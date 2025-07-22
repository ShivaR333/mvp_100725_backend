Updated Backend Reorganization Plan:

  backend/
  ├── data-input/              # Complete data processing pipeline
  │   ├── 01-upload-initiator-lambda.js
  │   ├── 02-map-schema-lambda.py
  │   ├── 03-clean-data-lambda.py
  │   ├── 04-save-data-lambda.py
  │   ├── 05-update-status-lambda.py      # ← Moved here
  │   ├── 06-get-dataset-status-lambda.js
  │   ├── 07-list-datasets-lambda.js
  │   ├── 08-get-data-preview-lambda.js
  │   ├── node_modules/
  │   ├── package.json
  │   ├── package-lock.json
  │   └── requirements.txt
  │
  ├── validation/              # Data validation functions
  │   ├── 01-validate-lambda.js  
  │   ├── 02-session-lambda.js
  │   ├── node_modules/
  │   ├── package.json
  │   └── package-lock.json
  │
  ├── orchestration/           # LLM agent orchestration (future)
  │   ├── 01-orchestrator-lambda.js      # ← This stays for now
  │   ├── node_modules/
  │   ├── package.json
  │   └── package-lock.json
  │
  ├── pipeline-setup/          # Empty for now
  │   └── .gitkeep
  │
  ├── train-and-inference/     # Empty for now  
  │   └── .gitkeep
  │
  ├── old/                     # Legacy files
  │   └── [existing app/ files]
  │
  └── README.md

  Logic:

  - data-input: Complete data processing pipeline from upload → status management
  - orchestration: Keeps the main orchestrator (which coordinates the Step Functions workflow) but will be primarily for future LLM agent orchestration
  - update-status: Now properly grouped with the data processing functions it supports

  This makes the data-input module truly self-contained for the entire CSV upload → processing → status workflow.

