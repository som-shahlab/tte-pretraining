import argparse
import datetime
import os
import pickle
import json
import numpy as np
from loguru import logger
from utils import LABELING_FUNCTIONS, load_data, save_data, return_df_jittered_time, save
import pandas as pd
from typing import Any, Callable, List, Optional, Set, Tuple
import collections
from abc import ABC, abstractmethod
import random
import femr
from femr.datasets import PatientDatabase
from femr.labelers.core import LabeledPatients
from femr.featurizers.core import FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from femr.labelers.core import NLabelsPerPatientLabeler, TimeHorizon
from femr.labelers.omop import (
    MortalityCodeLabeler,
)
import multiprocessing
from femr.extension import datasets as extension_datasets
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union, cast, Mapping
import femr.labelers
from femr import Patient
from femr.labelers.omop_inpatient_admissions import InpatientReadmissionLabeler

LabelType = Union[
    Literal["boolean"],
    Literal["numeric"],
    Literal["survival"],
    Literal["categorical"],
]

PH_codes = [
    a.strip()
    for a in """
1029634
I27.21
I27.22
416
416.1
416.2
I27.2
I27.29
I27.83
I27.0
416
416.0
416.8
I27.89
I27.82
1170535
I27.1
I27.20
I27.23
416.9
I27.81
I27.9
I27.9
I27.0
I27.1
67294
142308
I27.24
2065632
I27.2
""".split()
]

def identity(x: Any) -> Any:
    return x

# from utils import SurvivalValue, Label
from femr.labelers.core import SurvivalValue, Label

@dataclass
class Label:
    """An individual label for a particular patient at a particular time.
    The prediction for this label is made with all data <= time."""

    time: datetime.datetime
    value: Union[bool, int, float, str, SurvivalValue]

def get_death_concepts() -> List[str]:
    return [
        # "Death Type/OMOP generated",
        "Condition Type/OMOP4822053",
    ]

def get_inpatient_admission_concepts() -> List[str]:
    return ["Visit/IP"]

class Event:
    """An event within a patient record."""

    start: datetime.datetime
    code: str
    value: float | str | None

    # This class can also contain any number of other optional fields
    # Optional fields can be anything, but there are a couple that are considered "standard"
    #
    # - end: datetime, the end datetime for this event
    # - visit_id: int, the visit_id this event is tied to
    # - omop_table: str, the omop table this event was pulled from
    # - clarity_table: str, the clarity table where the event comes from

    def __init__(
        self,
        start: datetime.datetime,
        code: str,
        value: float | str | None = None,
        **kwargs: Any,
    ) -> None:
        self.start = start
        self.code = code
        self.value = value

        for a, b in kwargs.items():
            if b is not None:
                self.__dict__[a] = b

    def __getattr__(self, __name: str) -> Any:
        return None

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False

        def get_val(val: Any) -> Any:
            other = {}
            if val.__dict__ is not None:
                for a, b in val.__dict__.items():
                    if a not in ("code", "start", "value") and b is not None:
                        other[a] = b

            return (val.code, val.start, val.value, other)

        return bool(get_val(self) == get_val(other))

    def __repr__(self) -> str:
        val_str = ", ".join(f"{a}={b}" for a, b in self.__dict__.items())
        return f"Event({val_str})"

    def __getstate__(self) -> Dict[str, Any]:
        """Make this object pickleable (write)"""
        return self.__dict__

    def __setstate__(self, d: Dict[str, Any]) -> None:
        """Make this object pickleable (read)"""
        for a, b in d.items():
            self.__dict__[a] = b

@dataclass
class Patient:
    """A patient."""

    patient_id: int
    events: Sequence[Event]

def get_inpatient_admission_codes(
    ontology: extension_datasets.Ontology,
) -> Set[str]:
    # Don't get children here b/c it adds noise (i.e. "Medicare Specialty/AO")
    return set(get_inpatient_admission_concepts())

def get_inpatient_admission_events(patient: Patient, ontology: extension_datasets.Ontology) -> List[Event]:
    admission_codes: Set[str] = get_inpatient_admission_codes(ontology)
    events: List[Event] = []
    for e in patient.events:
        if e.code in admission_codes and e.omop_table == "visit_occurrence":
            # Error checking
            if e.start is None or e.end is None:
                raise RuntimeError(f"Event {e} cannot have `None` as its `start` or `end` attribute.")
            elif e.start > e.end:
                raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
            # Drop single point in time events
            if e.start == e.end:
                continue
            events.append(e)
    return events

def get_inpatient_admission_discharge_times(
    patient: Patient, ontology: extension_datasets.Ontology
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Return a list of all admission/discharge times for this patient."""
    events: List[Event] = get_inpatient_admission_events(patient, ontology)
    times: List[Tuple[datetime.datetime, datetime.datetime]] = []
    for e in events:
        if e.end is None:
            raise RuntimeError(f"Event {e} cannot have `None` as its `end` attribute.")
        if e.start > e.end:
            raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
        times.append((e.start, e.end))
    return times

class CodeLabeler(femr.labelers.TimeHorizonEventLabeler):
    def __init__(
        self, index_time_df, index_time_column, time_horizon, codes, ontology, offset=datetime.timedelta(days=0)
    ):
        self.codes = codes
        self.time_horizon = time_horizon

        self.prediction_times_map = collections.defaultdict(set)

        for _, row in index_time_df.iterrows():
            prediction_time: datetime.datetime = datetime.datetime.strptime(row[index_time_column], "%Y-%m-%d %H:%M:%S")
            self.prediction_times_map[row["PersonID"]].add(prediction_time - offset)

        super().__init__()

    def get_prediction_times(self, patient):
        return sorted(list(self.prediction_times_map[patient.patient_id]))

    def get_time_horizon(self):
        return self.time_horizon

    def allow_same_time_labels(self):
        return False

    def get_outcome_times(self, patient):
        outcome_times = set()

        for event in patient.events:
            if event.code in self.codes:
                outcome_times.add(event.start)

        return sorted(list(outcome_times))

class Labeler(ABC):
    """An interface for labeling functions.

    A labeling function applies a label to a specific datetime in a given patient's timeline.
    It can be thought of as generating the following list given a specific patient:
        [(patient ID, datetime_1, label_1), (patient ID, datetime_2, label_2), ... ]
    Usage:
    ```
        labeling_function: Labeler = Labeler(...)
        patients: Sequence[Patient] = ...
        labels: LabeledPatient = labeling_function.apply(patients)
    ```
    """

    @abstractmethod
    def label(self, patient: Patient) -> List[Label]:
        """Apply every label that is applicable to the provided patient.

        This is only called once per patient.

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list of :class:`Label` containing every label for the given patient
        """
        pass

    def get_patient_start_end_times(self, patient: Patient) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the (start, end) of the patient timeline.

        Returns:
            Tuple[datetime.datetime, datetime.datetime]: (start, end)
        """
        return (patient.events[0].start, patient.events[-1].start)

    @abstractmethod
    def get_labeler_type(self) -> LabelType:
        """Return what type of labels this labeler returns. See the Label class."""
        pass

    def apply(
        self,
        path_to_patient_database: Optional[str] = None,
        patients: Optional[Sequence[Patient]] = None,
        num_threads: int = 1,
        num_patients: Optional[int] = None,
        patient_ids: Optional[Set[int]] = None,
    ) -> LabeledPatients:
        """Apply the `label()` function one-by-one to each Patient in a sequence of Patients.

        Args:
            path_to_patient_database (str, optional): Path to `PatientDatabase` on disk.
                Must be specified if `patients = None`
            patients (Sequence[Patient], optional): An Sequence (i.e. list) of `Patient` objects.
                Must be specified if `path_to_patient_database = None`
                Typically this will be a `PatientDatabase` object.
            num_threads (int, optional): Number of CPU threads to parallelize across. Defaults to 1.
            num_patients (Optional[int], optional): Number of patients to process - useful for debugging.
                If specified, will take the first `num_patients` in the provided `PatientDatabase` / `patients` list.
                If None, use all patients.

        Returns:
            LabeledPatients: Maps patients to labels
        """
        if (patients is None and path_to_patient_database is None) or (
            patients is not None and path_to_patient_database is not None
        ):
            raise ValueError("Must specify exactly one of `patient_database` or `path_to_patient_database`")

        if path_to_patient_database:
            # Load patientdatabase if specified
            assert patients is None
            patient_database = PatientDatabase(path_to_patient_database)
            num_patients = len(patient_database) if not num_patients else num_patients
            pids = list(patient_database)
            patient_map = None
        else:
            # Use `patients` if specified
            assert patients is not None
            num_patients = len(patients) if not num_patients else num_patients
            patient_map = {p.patient_id: p for p in patients}
            pids = list(patient_map.keys())

        if patient_ids is not None:
            pids = [pid for pid in pids if pid in patient_ids]

        pids = pids[:num_patients]

        # Split patient IDs across parallelized processes
        pid_parts = np.array_split(pids, num_threads * 10)

        # NOTE: Super hacky workaround to pickling limitations
        if hasattr(self, "ontology") and isinstance(self.ontology, extension_datasets.Ontology):  # type: ignore
            # Remove ontology due to pickling, add it back later
            self.ontology: extension_datasets.Ontology = None  # type: ignore
        if (
            hasattr(self, "labeler")
            and hasattr(self.labeler, "ontology")
            and isinstance(self.labeler.ontology, extension_datasets.Ontology)
        ):
            # If NLabelsPerPatient wrapper, go to sublabeler and remove ontology due to pickling
            self.labeler.ontology: extension_datasets.Ontology = None  # type: ignore

        # Multiprocessing
        tasks = [(self, patient_map, path_to_patient_database, pid_part) for pid_part in pid_parts if len(pid_part) > 0]

        if num_threads != 1:
            ctx = multiprocessing.get_context("forkserver")
            with ctx.Pool(num_threads) as pool:
                results = []
                for res in pool.imap_unordered(_apply_labeling_function, tasks):
                    results.append(res)
        else:
            results = []
            for task in tasks:
                results.append(_apply_labeling_function(task))

        # Join results and return
        patients_to_labels: Dict[int, List[Label]] = dict(collections.ChainMap(*results))
        return LabeledPatients(patients_to_labels, self.get_labeler_type())


def _apply_labeling_function(
    args: Tuple[Labeler, Optional[Mapping[int, Patient]], Optional[str], List[int]]
) -> Dict[int, List[Label]]:
    """Apply a labeling function to the set of patients included in `patient_ids`.
    Gets called as a parallelized subprocess of the .apply() method of `Labeler`."""
    labeling_function: Labeler = args[0]
    patients: Optional[Mapping[int, Patient]] = args[1]
    path_to_patient_database: Optional[str] = args[2]
    patient_ids: List[int] = args[3]

    if path_to_patient_database is not None:
        patients = cast(Mapping[int, Patient], PatientDatabase(path_to_patient_database))

    # Hacky workaround for Ontology not being picklable
    if (
        hasattr(labeling_function, "ontology")  # type: ignore
        and labeling_function.ontology is None  # type: ignore
        and path_to_patient_database  # type: ignore
    ):  # type: ignore
        labeling_function.ontology = patients.get_ontology()  # type: ignore
    if (
        hasattr(labeling_function, "labeler")
        and hasattr(labeling_function.labeler, "ontology")
        and labeling_function.labeler.ontology is None
        and path_to_patient_database
    ):
        labeling_function.labeler.ontology = patients.get_ontology()  # type: ignore

    patients_to_labels: Dict[int, List[Label]] = {}
    for patient_id in patient_ids:
        patient: Patient = patients[patient_id]  # type: ignore
        labels: List[Label] = labeling_function.label(patient)
        patients_to_labels[patient_id] = labels

    return patients_to_labels

class TimeHorizonEventLabeler_TTE(Labeler):

    def __init__(
        self,
        index_time_df: str = None,
        outcome_time_df: str = None,
        prediction_time_adjustment_func: Callable = identity,
    ):
        self.index_time_df: pd.DataFrame = index_time_df
        self.outcome_time_df: pd.DataFrame = outcome_time_df

    @abstractmethod
    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        pass

    def get_outcome_times_from_csv(self, patient: Patient) -> List[datetime.datetime]:
        pass

    def get_prediction_times_from_csv(self, patient: Patient) -> List[datetime.datetime]:
        """Return prediction times based on a given CSV."""
        times: List[datetime.datetime] = []
        last_time = None
        # df = pd.read_csv(self.index_time_csv_path)
        df = self.index_time_df
        time_column = 'ordering_date'
        # df[time_column] = pd.to_datetime(df[time_column])
        df_patient = df[df["person_id"] == patient.patient_id]
        for _, row in df_patient.iterrows():
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(
                datetime.datetime.strptime(row[time_column], "%Y-%m-%d %H:%M:%S")
            )
            if last_time != prediction_time:
                times.append(prediction_time)
                last_time = prediction_time
        times = sorted(list(set(times)))
        return times

    @abstractmethod
    def get_time_horizon(self) -> TimeHorizon:
        pass

    @abstractmethod
    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        pass

    def get_patient_start_end_times(self, patient: Patient) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the datetimes that we consider the (start, end) of this patient."""
        return (patient.events[0].start, patient.events[-1].start)

    def get_labeler_type(self) -> LabelType:
        """Return boolean labels (TRUE if event occurs in TimeHorizon, FALSE otherwise)."""
        return "survival"

    def allow_same_time_labels(self) -> bool:
        """Whether or not to allow labels with events at the same time as prediction"""
        return True

    def label(self, patient: Patient) -> List[Label]:
        """Return a list of Labels for an individual patient.

        Assumes that events in `patient.events` are already sorted in chronologically
        ascending order (i.e. start -> end).

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list containing a label for each datetime returned by `get_prediction_times()`
        """
        if len(patient.events) == 0:
            return []

        __, end_time = self.get_patient_start_end_times(patient)
        if self.index_time_df is not None:
            prediction_times: List[datetime.datetime] = self.get_prediction_times_from_csv(patient)
        else:
            prediction_times: List[datetime.datetime] = self.get_prediction_times(patient)
        if self.outcome_time_df is not None:
            outcome_times: List[datetime.datetime] = self.get_outcome_times_from_csv(patient)
        else:
            outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)
        time_horizon: TimeHorizon = self.get_time_horizon()

        # Get (start, end) of time horizon. If end is None, then it's infinite (set timedelta to max)
        time_horizon_start: datetime.timedelta = time_horizon.start
        time_horizon_end: Optional[datetime.timedelta] = time_horizon.end  # `None` if infinite time horizon

        # For each prediction time, check if there is an outcome which occurs within the (start, end)
        # of the time horizon
        results: List[Label] = []
        curr_outcome_idx: int = 0
        last_time = None
        for time in prediction_times:
            if last_time is not None:
                assert time > last_time, f"Must be ascending prediction times, instead got {last_time} <= {time}"

            last_time = time
            # try:
            while curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] < time + time_horizon_start:
                # `curr_outcome_idx` is the idx in `outcome_times` that corresponds to the first
                # outcome EQUAL or AFTER the time horizon for this prediction time starts (if one exists)
                curr_outcome_idx += 1
            # except:
            #     pass

            if curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] == time:
                if not self.allow_same_time_labels():
                    continue
                warnings.warn(
                    "You are making predictions at the same time as the target outcome."
                    "This frequently leads to label leakage."
                )

            # TRUE if an event occurs within the time horizon
            is_outcome_occurs_in_time_horizon: bool = (
                (
                    # ensure there is an outcome
                    # (needed in case there are 0 outcomes)
                    curr_outcome_idx
                    < len(outcome_times)
                )
                and (
                    # outcome occurs after time horizon starts
                    time + time_horizon_start
                    <= outcome_times[curr_outcome_idx]
                )
                and (
                    # outcome occurs before time horizon ends (if there is an end)
                    (time_horizon_end is None)
                    or outcome_times[curr_outcome_idx] <= time + time_horizon_end
                )
            )
            # TRUE if patient is censored (i.e. timeline ends BEFORE this time horizon ends,
            # so we don't know if the outcome happened after the patient timeline ends)
            # If infinite time horizon labeler, then assume no censoring
            is_censored: bool = end_time < time + time_horizon_end if (time_horizon_end is not None) else False

            if is_outcome_occurs_in_time_horizon:
                results.append(Label(time=time, value=SurvivalValue(is_censored=False, time_to_event=outcome_times[curr_outcome_idx] - time)))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                if end_time - time < datetime.timedelta(seconds=0):
                    end_time = patient.events[-2].start
                assert end_time >= time, f"End time {end_time} must be >= prediction time {time}"
                results.append(Label(time=time, value=SurvivalValue(is_censored=True, time_to_event=end_time - time)))
            elif is_censored:
                # Censored => None
                if end_time - time < datetime.timedelta(seconds=0):
                    end_time = patient.events[-2].start
                assert end_time >= time, f"End time {end_time} must be >= prediction time {time}"
                results.append(Label(time=time, value=SurvivalValue(is_censored=True, time_to_event=end_time - time)))

        return results

class CodeLabeler_TTE(TimeHorizonEventLabeler_TTE):
    """Apply a label based on 1+ outcome_codes' occurrence(s) over a fixed time horizon."""

    def __init__(
        self,
        outcome_codes: List[str],
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Callable = identity,
        index_time_column: str = None,  # column name for index time
        index_time_df: pd.DataFrame = None,  # dataframe with index time
        outcome_time_column: str = None,  # column name for outcome time
        outcome_time_df: pd.DataFrame = None,  # dataframe with outcome time
    ):
        """Create a CodeLabeler, which labels events whose index in your Ontology is in `self.outcome_codes`

        Args:
            prediction_codes (List[int]): Events that count as an occurrence of the outcome.
            time_horizon (TimeHorizon): An interval of time. If the event occurs during this time horizon, then
                the label is TRUE. Otherwise, FALSE.
            prediction_codes (Optional[List[int]]): If not None, limit events at which you make predictions to
                only events with an `event.code` in these codes.
            prediction_time_adjustment_func (Optional[Callable]). A function that takes in a `datetime.datetime`
                and returns a different `datetime.datetime`. Defaults to the identity function.
        """
        self.outcome_codes: List[str] = outcome_codes
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_codes: Optional[List[str]] = prediction_codes
        self.prediction_time_adjustment_func: Callable = prediction_time_adjustment_func
        self.index_time_column: str = index_time_column
        self.index_time_df: pd.DataFrame = index_time_df
        self.outcome_time_column: str = outcome_time_column
        self.outcome_time_df: pd.DataFrame = outcome_time_df

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time (possibly modified by prediction_time_adjustment_func)
        as the time to make a prediction. Default to all events whose `code` is in `self.prediction_codes`."""
        times: List[datetime.datetime] = []
        last_time = None
        for e in patient.events:
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(e.start)
            if ((self.prediction_codes is None) or (e.code in self.prediction_codes)) and (
                last_time != prediction_time
            ):
                times.append(prediction_time)
                last_time = prediction_time
        return times

    def get_outcome_times_from_csv(self, patient: Patient) -> List[datetime.datetime]:
        """Return prediction times based on a given CSV."""
        times: List[datetime.datetime] = []
        last_time = None
        df = self.outcome_time_df
        time_column = self.outcome_time_column
        df_patient = df[df["person_id"] == patient.patient_id]
        for _, row in df_patient.iterrows():
            if type(row[time_column]) == str:
                outcome_time: datetime.datetime = self.prediction_time_adjustment_func(
                    datetime.datetime.strptime(row[time_column], "%Y-%m-%d %H:%M:%S")
                )
            elif type(row[time_column]) == datetime.datetime:
                outcome_time = self.prediction_time_adjustment_func(row[time_column])
            if last_time != outcome_time:
                times.append(outcome_time)
                last_time = outcome_time
        times = sorted(list(set(times)))
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.outcome_codes:
                times.append(event.start)
        return times

    def allow_same_time_labels(self) -> bool:
        # We cannot allow labels at the same time as the codes since they will generally be available as features ...
        return False

class PredefinedEventTimeCSVLabeler(CodeLabeler_TTE):
    """Apply a label for predefined times in a given CSV within the `time_horizon`.
    Make prediction at admission time.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        index_time_column: str = None,
        index_time_df: pd.DataFrame = None,
        outcome_codes: List = None,
        # prediction_codes: Optional[List[str]] = None,
        # prediction_time_adjustment_func: Callable = identity,
    ):
        # """Create a Mortality labeler."""
        # outcome_codes = list(
        #     femr.labelers.omop.map_omop_concept_codes_to_femr_codes(ontology, get_death_concepts(), is_ontology_expansion=True)
        # )

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            # prediction_codes=prediction_codes,
            # prediction_time_adjustment_func=prediction_time_adjustment_func,
            index_time_column=index_time_column,  # column name for index time
            index_time_df=index_time_df,  # dataframe containing index time
        )


class SourceCodeLabeler(femr.labelers.TimeHorizonEventLabeler):
    def __init__(self, index_time_df, index_time_column, time_horizon, codes, ontology):
        self.codes = codes
        self.time_horizon = time_horizon

        self.prediction_times_map = collections.defaultdict(set)

        for _, row in index_time_df.iterrows():
            prediction_time: datetime.datetime = datetime.datetime.strptime(row[index_time_column], "%Y-%m-%d %H:%M:%S")
            self.prediction_times_map[row["PersonID"]].add(prediction_time)

        super().__init__()

    def get_prediction_times(self, patient):
        return sorted(list(self.prediction_times_map[patient.patient_id]))

    def get_time_horizon(self):
        return self.time_horizon

    def allow_same_time_labels(self):
        return False

    def get_outcome_times(self, patient):
        outcome_times = set()

        for event in patient.events:
            if (
                event.omop_table == "condition_occurrence"
                and event.source_code is not None
                and event.source_code in self.codes
            ):
                outcome_times.add(event.start)

        return sorted(list(outcome_times))


class SourceCodeLabeler_TTE(TimeHorizonEventLabeler_TTE):
    def __init__(self, index_time_df, index_time_column, time_horizon, codes, ontology):
        self.codes = codes
        self.time_horizon = time_horizon

        self.prediction_times_map = collections.defaultdict(set)

        for _, row in index_time_df.iterrows():
            prediction_time: datetime.datetime = datetime.datetime.strptime(row[index_time_column], "%Y-%m-%d %H:%M:%S")
            self.prediction_times_map[row["person_id"]].add(prediction_time)

        super().__init__()

    def get_prediction_times(self, patient):
        return sorted(list(self.prediction_times_map[patient.patient_id]))

    def get_time_horizon(self):
        return self.time_horizon

    def allow_same_time_labels(self):
        return False

    def get_outcome_times(self, patient):
        outcome_times = set()

        for event in patient.events:
            if (
                event.omop_table == "condition_occurrence"
                and event.source_code is not None
                and event.source_code in self.codes
            ):
                outcome_times.add(event.start)

        return sorted(list(outcome_times))


class ModifiedReadmission(InpatientReadmissionLabeler):
    def __init__(self, index_time_df, index_time_column, time_horizon, ontology):
        self.prediction_times_map = collections.defaultdict(set)

        for _, row in index_time_df.iterrows():
            prediction_time: datetime.datetime = datetime.datetime.strptime(row[index_time_column], "%Y-%m-%d %H:%M:%S")
            self.prediction_times_map[row["person_id"]].add(prediction_time)

        super().__init__(ontology, time_horizon)

    def get_prediction_times(self, patient):
        return sorted(list(self.prediction_times_map[patient.patient_id]))

    def allow_same_time_labels(self):
        return False

class InpatientReadmissionLabeler_TTE(TimeHorizonEventLabeler_TTE):
    """
    This labeler is designed to predict whether a patient will be readmitted within `time_horizon`
    It explicitly does not try to deal with categorizing admissions as "unexpected" or not and is thus
    not comparable to other work.

    Prediction time: At discharge from an inpatient admission. Defaults to shifting prediction time
                     to the end of the day.
    Time horizon: Interval of time after discharg of length `time_horizon`
    Label: TRUE if patient has an inpatient admission within `time_horizon`

    Defaults to 30-day readmission labeler,
        i.e. `time_horizon = TimeHorizon(1 second, 30 days)`
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon = TimeHorizon(
            start=datetime.timedelta(seconds=1), end=datetime.timedelta(days=30)
        ),  # type: ignore
        prediction_time_adjustment_func: Callable = identity,  # move_datetime_to_end_of_day,
        index_time_csv_path: str = None,  # read in index time from csv
        index_time_column: str = None,  # column name for index time
        index_time_df: pd.DataFrame = None,  # dataframe with index time
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_time_adjustment_func = prediction_time_adjustment_func  # prediction_time_adjustment_func
        self.index_time_csv_path = index_time_csv_path
        self.index_time_column = index_time_column
        self.index_time_df = index_time_df
        self.outcome_time_df = None

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions."""
        times: List[datetime.datetime] = []
        for admission_time, __ in get_inpatient_admission_discharge_times(patient, self.ontology):
            times.append(admission_time)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return end of admission as prediction timm."""
        times: List[datetime.datetime] = []
        prev_discharge_date: datetime.date = datetime.date(1900, 1, 1)

        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(discharge_time)

            # Ignore patients who are readmitted the same day they were discharged b/c of data leakage
            if admission_time.date() <= prev_discharge_date:
                continue

            times.append(prediction_time)
            prev_discharge_date = discharge_time.date()
        times = sorted(list(set(times)))
        return times

    def get_prediction_times_from_csv(self, patient: Patient) -> List[datetime.datetime]:
        """Return prediction times based on a given CSV."""
        times: List[datetime.datetime] = []
        last_time = None
        # df = pd.read_csv(self.index_time_csv_path)
        df = self.index_time_df
        time_column = self.index_time_column
        # df[time_column] = pd.to_datetime(df[time_column])
        df_patient = df[df["person_id"] == patient.patient_id]
        for _, row in df_patient.iterrows():
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(
                datetime.datetime.strptime(row[time_column], "%Y-%m-%d %H:%M:%S")
            )
            if last_time != prediction_time:
                times.append(prediction_time)
                last_time = prediction_time
        times = sorted(list(set(times)))
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

class ModifiedReadmission_TTE(InpatientReadmissionLabeler_TTE):
    def __init__(self, index_time_df, index_time_column, time_horizon, ontology):
        self.prediction_times_map = collections.defaultdict(set)

        for _, row in index_time_df.iterrows():
            prediction_time: datetime.datetime = datetime.datetime.strptime(row[index_time_column], "%Y-%m-%d %H:%M:%S")
            self.prediction_times_map[row["person_id"]].add(prediction_time)

        super().__init__(ontology, time_horizon)

    def get_prediction_times(self, patient):
        return sorted(list(self.prediction_times_map[patient.patient_id]))

    def allow_same_time_labels(self):
        return False


def stripoff_subminute(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)


def save_labeled_patients_to_csv(labeled_patients: LabeledPatients, path_to_csv: str) -> pd.DataFrame:
    """Converts a LabeledPatient object -> pd.DataFrame and saves as CSV to `path_to_csv`"""
    rows = []
    for patient, labels in labeled_patients.items():
        omop_patient_id = patient  # for some reason the pipeline uses the OMOP ID for labelers as the patient ID
        for l in labels:
            if type(l.value) == type(False):
                rows.append((omop_patient_id, l.time, l.value, "boolean"))
            elif type(l.value) == type("False"):
                rows.append((omop_patient_id, l.time, l.value, "str"))
    df = pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "prediction_time",
            "value",
            "label_type",
        ],
    )
    df.to_csv(path_to_csv, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr featurization")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to femr database")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to save labeles and featurizers")
    parser.add_argument(
        "--labeling_function",
        required=True,
        type=str,
        help="Name of labeling function to create.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )
    parser.add_argument(
        "--max_labels_per_patient",
        type=int,
        help="Max number of labels to keep per patient (excess labels are randomly discarded)",
        default=None,
    )
    parser.add_argument(
        "--path_to_chexpert_csv",
        type=str,
        help="Path to chexpert labeled csv file. Specific to chexpert labeler",
        default=None,
    )
    parser.add_argument("--is_skip_label", action="store_true", help="If specified, skip labeling step", default=False)
    parser.add_argument("--is_skip_featurize", action="store_true", help="If specified, skip featurize step", default=False)
    parser.add_argument(
        "--index_time_csv_path",
        type=str,
        help="path to a csv with desired index times",
        default=None,
    )
    parser.add_argument(
        "--index_time_column",
        type=str,
        help="column name in index_time_csv_path that contains the index times",
        default=None,
    )
    parser.add_argument(
        "--outcome_time_csv_path",
        type=str,
        help="path to a csv with desired outcome times",
        default=None,
    )
    parser.add_argument(
        "--outcome_time_column",
        type=str,
        help="column name in outcome_time_csv_path that contains the outcome times",
        default=None,
    )
    parser.add_argument(
        "--outcome_label_column",
        type=str,
        help="column name in outcome_time_csv_path that contains the outcome label",
        default=None,
    )
    parser.add_argument(
        "--secret",
        type=int,
        help="secret seed",
        default=91108281998,
    )
    parser.add_argument(
        "--path_to_PHI_database",
        type=str,
        default="omop_cdm5_subset_2023_05_06_extract_no_observation_v2",
        help="Path to femr PHI database for all",
    )
    parser.add_argument(
        "--path_to_AnonMapping",
        type=str,
        default=None,
        help="Path to map person_id to anon id",
    )

    args = parser.parse_args()

    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads
    MAX_LABELS_PER_PATIENT: int = args.max_labels_per_patient

    PATH_TO_OUTPUT_DIR = os.path.join(PATH_TO_OUTPUT_DIR, args.labeling_function)

    # Logging
    path_to_log_file: str = os.path.join(PATH_TO_OUTPUT_DIR, "info.log")
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")  # connect logger to file
    logger.info(f"Labeling function: {args.labeling_function}")
    logger.info(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    logger.info(f"Saving output to: {PATH_TO_OUTPUT_DIR}")
    logger.info(f"Max # of labels per patient: {MAX_LABELS_PER_PATIENT}")
    logger.info(f"# of threads: {NUM_THREADS}")
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # create directories to save files
    PATH_TO_SAVE_ID_LABELED_PATIENTS: str = os.path.join(PATH_TO_OUTPUT_DIR, "labeled_patients_identified.csv")
    PATH_TO_SAVE_LABELED_PATIENTS: str = os.path.join(PATH_TO_OUTPUT_DIR, "labeled_patients.csv")
    PATH_TO_SAVE_PREPROCESSED_FEATURIZERS: str = os.path.join(PATH_TO_OUTPUT_DIR, "preprocessed_featurizers.pkl")
    PATH_TO_SAVE_FEATURIZED_PATIENTS: str = os.path.join(PATH_TO_OUTPUT_DIR, "featurized_patients.pkl")
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Load PatientDatabase + Ontology
    logger.info(f"Start | Load PatientDatabase")
    database = PatientDatabase(PATH_TO_PATIENT_DATABASE)
    id_database = PatientDatabase(args.path_to_PHI_database)
    ontology = database.get_ontology()
    id_ontology = id_database.get_ontology()
    logger.info(f"Finish | Load PatientDatabase")

    # Load index times from csv
    # change patient id to anon id
    # change index time/outcome time to jittered time

    index_time_df = pd.read_csv(args.index_time_csv_path)
    if args.path_to_AnonMapping is not None:
        AnonMapping = load_data(args.path_to_AnonMapping)
    else:
        AnonMapping = None

    patient_ids = set(list(set(index_time_df["person_id"])))

    mortality_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            id_ontology, femr.labelers.omop.get_death_concepts(), is_ontology_expansion=True
        )
    )
    precise_ph_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            id_ontology, {"SNOMED/70995007"}, is_ontology_expansion=True
        )
    )
    precise_pe_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            id_ontology, {"SNOMED/59282003"}, is_ontology_expansion=True
        )
    )
    Atelectasis_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            id_ontology, {"SNOMED/46621007"}, is_ontology_expansion=True
        )
    )
    Cardiomegaly_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            id_ontology, {"SNOMED/8186001"}, is_ontology_expansion=True
        )
    )
    Consolidation_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            id_ontology, {"SNOMED/95436008"}, is_ontology_expansion=True
        )
    )
    Edema_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            id_ontology, {"SNOMED/267038008"}, is_ontology_expansion=True
        )
    )
    Pleural_Effusion_codes = list(
        femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
            id_ontology, {"SNOMED/60046008"}, is_ontology_expansion=True
        )
    )
    if args.labeling_function == "tte_mortality":
        labeler = PredefinedEventTimeCSVLabeler(
            id_database.get_ontology(),
            TimeHorizon(datetime.timedelta(minutes=1), None),
            args.index_time_column,
            index_time_df,
            mortality_codes,
        )
    elif args.labeling_function == "tte_readmission":
        labeler = ModifiedReadmission_TTE(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), None),
            id_database.get_ontology(),
        )
    elif args.labeling_function == "12_month_mortality":
        labeler = CodeLabeler(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=365)),
            mortality_codes,
            id_database.get_ontology(),
        )
    elif args.labeling_function == "6_month_mortality":
        labeler = CodeLabeler(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=180)),
            mortality_codes,
            id_database.get_ontology(),
        )
    elif args.labeling_function == "1_month_mortality":
        labeler = CodeLabeler(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=30)),
            mortality_codes,
            id_database.get_ontology(),
        )
    elif args.labeling_function == "1_month_readmission":
        labeler = ModifiedReadmission(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=30)),
            id_database.get_ontology(),
        )
    elif args.labeling_function == "6_month_readmission":
        labeler = ModifiedReadmission(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=180)),
            id_database.get_ontology(),
        )
    elif args.labeling_function == "12_month_readmission":
        labeler = ModifiedReadmission(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=365)),
            id_database.get_ontology(),
        )
    elif args.labeling_function == "12_month_PH":
        labeler = SourceCodeLabeler(
            index_time_df,
            args.index_time_column,
            femr.labelers.TimeHorizon(datetime.timedelta(days=1), datetime.timedelta(days=365)),
            PH_codes,
            id_database.get_ontology(),
        )
    elif args.labeling_function == "tte_PH":
        labeler = SourceCodeLabeler_TTE(
            index_time_df,
            args.index_time_column,
            femr.labelers.TimeHorizon(datetime.timedelta(days=1), None),
            PH_codes,
            id_database.get_ontology(),
        )
    elif args.labeling_function == "tte_Atelectasis":
        labeler = CodeLabeler_TTE(
            Atelectasis_codes,
            femr.labelers.TimeHorizon(datetime.timedelta(days=1), None),
            index_time_df=index_time_df,
            index_time_column=args.index_time_column,
            # id_database.get_ontology(),
        )
    elif args.labeling_function == "tte_Cardiomegaly":
        labeler = CodeLabeler_TTE(
            Cardiomegaly_codes,
            femr.labelers.TimeHorizon(datetime.timedelta(days=1), None),
            index_time_df=index_time_df,
            index_time_column=args.index_time_column,
            # id_database.get_ontology(),
        )
    elif args.labeling_function == "tte_Consolidation":
        labeler = CodeLabeler_TTE(
            Consolidation_codes,
            femr.labelers.TimeHorizon(datetime.timedelta(days=1), None),
            index_time_df=index_time_df,
            index_time_column=args.index_time_column,
            # id_database.get_ontology(),
        )
    elif args.labeling_function == "tte_Edema":
        labeler = CodeLabeler_TTE(
            Edema_codes,
            femr.labelers.TimeHorizon(datetime.timedelta(days=1), None),
            index_time_df=index_time_df,
            index_time_column=args.index_time_column,
            # id_database.get_ontology(),
        )
    elif args.labeling_function == "tte_Pleural_Effusion":
        labeler = CodeLabeler_TTE(
            Pleural_Effusion_codes,
            femr.labelers.TimeHorizon(datetime.timedelta(days=1), None),
            index_time_df=index_time_df,
            index_time_column=args.index_time_column,
            # id_database.get_ontology(),
        )
    elif args.labeling_function == "12_month_precise_PH":
        labeler = CodeLabeler(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=365)),
            precise_ph_codes,
            id_database.get_ontology(),
        )
    elif args.labeling_function == "PE":
        labeler = None
        labels = collections.defaultdict(list)
        for _, row in index_time_df.iterrows():
            labels[row["PersonID"]].append(
                femr.labelers.Label(
                    time=pd.to_datetime(row["ProcedureDatetime"]).to_pydatetime() - datetime.timedelta(days=1),
                    value=row["pe_positive_nlp"],
                )
            )

        for _, v in labels.items():
            v.sort(key=lambda a: a.time)

        id_labeled_patients = femr.labelers.LabeledPatients(labels, "boolean")
    elif args.labeling_function == "Cardiomegaly":
        labeler = None
        labels = collections.defaultdict(list)
        for _, row in index_time_df.iterrows():
            label_value = True if row["Cardiomegaly"] == 1 else False
            labels[row["person_id"]].append(
                femr.labelers.Label(
                    time=pd.to_datetime(row["StudyDate"]).to_pydatetime() - datetime.timedelta(days=1),
                    value=label_value,
                )
            )

        for _, v in labels.items():
            v.sort(key=lambda a: a.time)

        id_labeled_patients = femr.labelers.LabeledPatients(labels, "boolean")

    elif args.labeling_function == "PE_code":
        labeler = CodeLabeler(
            index_time_df,
            args.index_time_column,
            TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=30)),
            precise_pe_codes,
            id_database.get_ontology(),
            offset=datetime.timedelta(days=1),
        )

    elif args.labeling_function == "chexpert":
        assert args.path_to_chexpert_csv is not None, f"path_to_chexpert_csv cannot be {args.path_to_chexpert_csv}"
        labeler = ChexpertLabeler(args.path_to_chexpert_csv)
    else:
        raise ValueError(
            f"Labeling function `{args.labeling_function}` not supported. Must be one of: {LABELING_FUNCTIONS}."
        )

    # Determine how many labels to keep per patient
    if labeler is not None and args.max_labels_per_patient is not None and args.labeling_function != "chexpert":
        labeler = NLabelsPerPatientLabeler(labeler, seed=0, num_labels=MAX_LABELS_PER_PATIENT)

    if args.is_skip_label:
        logger.critical(f"Skipping labeling step. Loading labeled patients from @ {PATH_TO_SAVE_LABELED_PATIENTS}")
        labeled_patients = pickle.load(open(PATH_TO_SAVE_LABELED_PATIENTS, "rb"))
    else:
        logger.info(f"Start | Label {len(patient_ids)} patients")

        if labeler is not None:
            id_labeled_patients = labeler.apply(
                path_to_patient_database=args.path_to_PHI_database,
                num_threads=NUM_THREADS,
                patient_ids=patient_ids,
            )
        id_labeled_patients.save(PATH_TO_SAVE_ID_LABELED_PATIENTS)
        # save(id_labeled_patients, PATH_TO_SAVE_ID_LABELED_PATIENTS)
        logger.info("Finish | Label patients")
        logger.info(
            "LabeledPatient stats:\n"
            f"Total # of patients = {id_labeled_patients.get_num_patients()}\n"
            f"Total # of patients with at least one label = {id_labeled_patients.get_num_patients()}\n"
            f"Total # of labels = {id_labeled_patients.get_num_labels()}"
        )

        shifted_label_map = {}

        for id_pid, id_labels in id_labeled_patients.items():
            if AnonMapping is not None:
                deid_pid = AnonMapping[id_pid]
            else:
                deid_pid = id_pid
            offset = database.get_patient_birth_date(deid_pid) - id_database.get_patient_birth_date(id_pid)
            deid_labels = []
            for id_label in id_labels:
                deid_label = femr.labelers.Label(time=id_label.time + offset, value=id_label.value)
                deid_labels.append(deid_label)
            shifted_label_map[deid_pid] = deid_labels

        labeled_patients = femr.labelers.LabeledPatients(
            shifted_label_map, labeler_type=id_labeled_patients.get_labeler_type()
        )
        labeled_patients.save(PATH_TO_SAVE_LABELED_PATIENTS)
        # save(labeled_patients, PATH_TO_SAVE_LABELED_PATIENTS)

    if not args.is_skip_featurize:
        # Lets use both age and count featurizer
        age = AgeFeaturizer()
        count = CountFeaturizer(is_ontology_expansion=True)
        featurizer_age_count = FeaturizerList([age, count])

        # Preprocessing the featurizers, which includes processes such as normalizing age.
        logger.info("Start | Preprocess featurizers")
        featurizer_age_count.preprocess_featurizers(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
        save_data(featurizer_age_count, PATH_TO_SAVE_PREPROCESSED_FEATURIZERS)
        logger.info("Finish | Preprocess featurizers")

        logger.info("Start | Featurize patients")
        results = featurizer_age_count.featurize(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
        save_data(results, PATH_TO_SAVE_FEATURIZED_PATIENTS)
        logger.info("Finish | Featurize patients")
        feature_matrix, patient_ids, label_values, label_times = (
            results[0],
            results[1],
            results[2],
            results[3],
        )
        label_set, counts_per_label = np.unique(label_values, return_counts=True)
        logger.info(
            "FeaturizedPatient stats:\n"
            f"feature_matrix={repr(feature_matrix)}\n"
            f"patient_ids={repr(patient_ids)}\n"
            f"label_values={repr(label_values)}\n"
            f"label_set={repr(label_set)}\n"
            f"counts_per_label={repr(counts_per_label)}\n"
            f"label_times={repr(label_times)}"
        )

    with open(os.path.join(PATH_TO_OUTPUT_DIR, "done.txt"), "w") as f:
        f.write("done")

    logger.info("Done!")
