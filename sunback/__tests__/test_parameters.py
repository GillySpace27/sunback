from unittest import TestCase

import parameters


class TestParameters(TestCase):

    def real_number_test(self, func):
        self.assertFalse(func(1))
        self.assertFalse(func(100000))

        self.assertRaises(AssertionError, func, 0)
        self.assertRaises(AssertionError, func, -1)
        self.assertRaises(AssertionError, func, 'f')

    def test_check_real_number(self):
        self.real_number_test(parameters.Parameters().check_real_number)

    def test_set_default_values(self):
        self.assertFalse(parameters.Parameters().set_default_values())

    def test_set_update_delay_seconds(self):
        self.real_number_test(parameters.Parameters().set_delay_seconds)

    def test_set_time_multiplier(self):
        self.real_number_test(parameters.Parameters().set_time_multiplier)



    # def test_set_local_directory(self):
    #     self.fail()
    #
    # def test_set_wavelengths(self):
    #     self.fail()

    def test_set_download_resolution(self):
        self.real_number_test(parameters.Parameters().set_download_resolution)

    # def test_set_web_image_frame(self):
    #     self.fail()
    #
    # def test_set_file_ending(self):
    #     self.fail()
    #
    # def test_make_web_paths(self):
    #     self.fail()
    #
    # def test_append_to_web_paths(self):
    #     self.fail()
    #
    # def test_has_all_necessary_data(self):
    #     self.fail()
    #
    # def test_get_local_path(self):
    #     self.fail()
    #
    # def test_discover_best_default_directory(self):
    #     self.fail()
    #
    # def test_sleep_for_time(self):
    #     self.fail()