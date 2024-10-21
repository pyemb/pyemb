c = get_config()

# Ignore cells with the "nbconvert-remove" tag
c.TagRemovePreprocessor.remove_cell_tags = {"nbconvert-remove"}
c.TagRemovePreprocessor.enabled = True


c.RSTExporter.figure_format = 'svg'
c.RSTExporter.figure_dpi = 300 