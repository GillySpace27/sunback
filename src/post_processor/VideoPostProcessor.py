from post_processor.PostProcessor import PostProcessor


class VideoPostProcessor(PostProcessor):
    def __init__(self):
        pass
    
    def process(self):
        raise NotImplementedError()


        # self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(self.this_name, self.beginTime, '{}'))
    
        # name = "{}_{}".format(self.this_name, "max")
        # self.soni = Sonifier(self.params, self.save_path, name, self.video_name_stem, frames_per_second=self.params.frames_per_second())
    
        # print("\nMovie: {}".format(self.this_name))