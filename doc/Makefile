MD_FILES = $(shell ls *.md)
DOT_FILES = $(shell ls images/*.dot)

ALL_FILES = $(MD_FILES) \
			$(DOT_FILES)

html: $(ALL_FILES)
	doxygen
