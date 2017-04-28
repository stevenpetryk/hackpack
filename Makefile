all:
		pandoc hackpack.java.md -o hackpack.html --no-highlight
		make test

test:
		./run-md hackpack.java.md
