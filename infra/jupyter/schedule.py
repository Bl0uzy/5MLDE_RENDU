import config
from orchestration import complete_ml

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import (
    CronSchedule,
)

modeling_deployment_every_sunday = Deployment.build_from_flow(
    name="Model training Deployment",
    flow=complete_ml,
    version="1.0",
    tags=["model"],
    schedule=CronSchedule(cron="0 0 * * 0"),
    parameters={
        "train_path": config.TRAIN_DATA,
        "test_path": config.TEST_DATA,
    }
)

if __name__ == "__main__":
    modeling_deployment_every_sunday.apply()