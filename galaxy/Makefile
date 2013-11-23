COMPILER=python
FLAGS=source/setup.py build_ext --inplace
main:
	$(COMPILER) $(FLAGS)
	if [ -d build ]; then rm -r build; fi
clean:
	rm optimized_functions.so
