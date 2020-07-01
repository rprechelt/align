##
# ##
# align
#
# @file
# @version 0.0.1

# find python3
PYTHON=`/usr/bin/which python3`

# our testing targets
.PHONY: tests flake black isort all

all: mypy isort black flake tests

tests:
	${PYTHON} -m pytest --cov=align tests

flake:
	${PYTHON} -m flake8 align

black:
	${PYTHON} -m black -t py37 align tests

isort:
	${PYTHON} -m isort --atomic -rc -y align tests

mypy:
	${PYTHON} -m mypy align

# end
