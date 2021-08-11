# Copyright (c) 2021, Justin Kim

# We are making my contributions/submissions to this project solely in our
# personal capacity and are not conveying any rights to any intellectual
# property of any third parties.

import os
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = 'Options for testing')
	parser.add_argument('-r', '--repeat', type=int, help='Number of times to repeat the test')
	parser.add_argument('-gen_old', '--gen_old_arch', action='store_true', help='True if generating for old architectures')
	parser.add_argument('-d', '--debug', action='store_true', help = 'Builds in debug mode')
	parser.add_argument('-c', '--clear', action='store_true', help = 'Clears cache before building')

	args= parser.parse_args()

	home_dir = os.getcwd()

	if args.clear:
		print("Cleaning Cache")
		os.system("rm CMakeCache.txt")

	cmake_str = "cmake"

	if args.debug:
		print("Building in Debug mode")
		cmake_str += " -DCMAKE_BUILD_TYPE=Debug"
	if args.gen_old_arch:
		print("Building for old arch")
		cmake_str += " -DGEN_OLD_ARCH=ON"
	
	cmake_str += " ."

	os.system(cmake_str)
	os.system("make -j${nproc}")

	repeat_num = 1
	if args.repeat is not None :
		repeat_num = args.repeat

	num_failures = 0
	for i in range(0, repeat_num):
		print("repeating  for : {n}".format(n = i))
		rtn_code = os.system("./bin/UnitTests")>>8
		if rtn_code != 0:
			print("failure!")
			num_failures += 1

	if num_failures == 0:
		print("All tests were successful")
	else:
		print("Test failed for {0} times".format(num_failures))

