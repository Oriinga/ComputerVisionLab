import os
import wandb
import logging

logging.basicConfig(level=logging.DEBUG)
wandb.init(project="your_project_name")

os.environ['WANDB_DEBUG'] = 'true'
os.environ["WANDB_API_URL"] = "https://api.wandb.ai"
wandb.login()
wandb.init()
try:
    run = wandb.init(project="test_project", name="test_run", job_type="test",settings=wandb.Settings(start_method="fork"))
    print(f"Run initialized successfully. Run URL: {run.get_url()}")
except Exception as e:
    print(f"Failed to initialize W&B run. Error: {e}")
finally:
    if wandb.run:
        wandb.finish()