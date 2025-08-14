# preprocessing/spikewrap_pipeline.py
'''
Script to run spikewrap

'''
import spikewrap as sw
import os

def run_spikewrap(derivatives_base, subject_path, session_name):
    """
    Function runs spikewrap
    Input:
    derivatives_base: path to derivatives folder
    subject_path: path with ephys data
    session_name: name of this session

    Creates:
    Binned data (as a raw file)
    Kilosort4 output

    Currently runs kilosort4
    Settings are set so that kilosort4 does not do drift correction
    and does not do CAR (common average reference) on the data.
    """


    session = sw.Session(
        subject_path=subject_path,
        session_name=session_name,
        file_format="spikeglx",
        run_names="all",
        output_path = derivatives_base,
    )
    """
    session.preprocess(
        configs="neuropixels+kilosort2_5",
        per_shank=False,
        concat_runs=True,
    )


    #plots = session.plot_preprocessed(time_range=(0, 0.1), show=True)

    # you could not save here
    session.save_preprocessed(
        overwrite=True,
        n_jobs=1,
        slurm=False
    )
   """
    cfg = sw.load_config_dict(sw.get_configs_path() / "neuropixels+kilosort2_5.yaml")
    del cfg["sorting"]["kilosort2_5"]
  #  settings = {"n_chan_bin": 384, "nblocks": 0 , "highpass_cutoff": 100}  # nblocks=0 turns off drift correction, I think you can also do it with a "do_correction" (spikeinterface)
    #         "use_binary_file": True,
    #    "delete_recording_dat": True,
    cfg["sorting"]["kilosort4"] = {"do_CAR": False, "save_preprocessed_copy": True, "nblocks": 0 , "highpass_cutoff": 100}
    session.sort(cfg, run_sorter_method="local", per_shank=False, concat_runs=False)