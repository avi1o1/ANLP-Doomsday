# Shared colours, progress lines and the self-documenting help target.
#
# Include from any Makefile in the repository, after setting PROJECT:
#
#     PROJECT := ANLP Doomsday
#     include make/common.mk
#
# Document a target by putting `## description` on its target line, and group
# targets by putting `##@ Group name` on a line of its own. `make help` reads both
# out of the Makefile, so a new target is listed the moment it is written.
#
# Colour is on unless NO_COLOR is set or TERM is dumb, which covers pipes into
# files, CI logs and editor terminals that do not handle escapes.

ifeq ($(strip $(NO_COLOR))$(filter dumb,$(TERM)),)
BOLD  := \033[1m
DIM   := \033[2m
RED   := \033[31m
GREEN := \033[32m
AMBER := \033[33m
BLUE  := \033[34m
CYAN  := \033[36m
RESET := \033[0m
endif

# Progress lines, used as: @$(STEP) "what is starting"
STEP := printf '$(BLUE)==>$(RESET) %s\n'
DONE := printf '$(GREEN) ok $(RESET) %s\n'
WARN := printf '$(AMBER) !! $(RESET) %s\n'
FAIL := printf '$(RED)fail$(RESET) %s\n'

# Sub-makes are called for their output, not their bookkeeping.
MAKEFLAGS += --no-print-directory

# A bare `make` lists what is available rather than starting a build, since the
# targets here range from a two-second LaTeX pass to a full experiment run.
.DEFAULT_GOAL := help

##@ Help

.PHONY: help
help: ## Show this message
	@printf '\n$(BOLD)%s$(RESET)\n' "$(PROJECT)"
	@printf '$(DIM)%s$(RESET)\n' "usage: make <target>"
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5); next } \
		/^[a-zA-Z0-9_%.-]+:.*##/ { \
			pad = 16 - length($$1); if (pad < 1) pad = 1; \
			printf "  $(CYAN)%s$(RESET)%*s%s\n", $$1, pad, "", $$2 }' \
		$(MAKEFILE_LIST)
	@printf '\n'
