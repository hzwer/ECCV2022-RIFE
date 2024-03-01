import nuke

# Force loading of the Inference node due to a bug in Nuke 13 - std::bad_alloc
nuke.load("Inference")
