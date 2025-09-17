import numpy as np
import spikewrap as sw 


subject_path = sw.get_example_data_path() / "rawdata" / "sub-001"

session = sw.Session(
    subject_path=subject_path,
    session_name="ses-001",
    file_format="spikeglx",  # or "openephys"
    run_names="all",
    probe=None,  # optional argument to set probe
)
