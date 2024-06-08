#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = image_classification_trash
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Define the Python files and directories to be checked
SRC_PYTHON_FILES = ./src/*.py
DATA_PYTHON_FILES = ./data/*.py
TEST_PYTHON_FILES = ./tests/*.py

## Install Python Dependencies
# .PHONY: requirements
# requirements:
# 	pipenv install
	



## Delete all compiled Python files
# .PHONY: clean
# clean:
# 	find . -type f -name "*.py[co]" -delete
# 	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 $(SRC_PYTHON_FILES)
	flake8 $(DATA_PYTHON_FILES)
	flake8 $(TEST_PYTHON_FILES)

## Format source code with black
.PHONY: format
format:
	black $(SRC_PYTHON_FILES)
	black $(DATA_PYTHON_FILES)
	black $(TEST_PYTHON_FILES)




## Set up python interpreter environment
# .PHONY: create_environment
# create_environment:
# 	pipenv --python $(PYTHON_VERSION)
# 	@echo ">>> New pipenv created. Activate with:\npipenv shell"
	



