#!/bin/bash

plantcv-workflow.py --config multi_plant_workflow_config.json

plantcv-utils.py json2csv --json multi_plant_results.json

