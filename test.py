from indexing.training_img_embedding import fine_tune_openclip_from_ftimages
fine_tune_openclip_from_ftimages("videoguide", lambda p,s,i: print(p, s, i))