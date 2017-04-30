all:
	pandoc hackpack.java.md -o hackpack.html --no-highlight --katex
	make test

test:
	./run-md hackpack.java.md
