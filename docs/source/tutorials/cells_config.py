c = get_config()

# Ignore cells with the "nbconvert-remove" tag
c.TagRemovePreprocessor.remove_cell_tags = {"nbconvert-remove"}
c.TagRemovePreprocessor.enabled = True
