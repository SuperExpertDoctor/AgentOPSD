from pathlib import Path


def test_tensorboard_tracking_writes_event_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))

    from verl.utils.tracking import Tracking

    tracking = Tracking(
        project_name="test-project",
        experiment_name="test-tensorboard",
        default_backend=["tensorboard"],
    )
    tracking.log({"train/loss": 0.5}, step=1)
    tracking.logger["tensorboard"].writer.flush()
    tracking.logger["tensorboard"].finish()

    assert list(Path(tmp_path).glob("events.out.tfevents.*"))
