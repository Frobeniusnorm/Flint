AHTML := $(wildcard *.ahtml)
all: $(AHTML) generator/SideGenerator
	generator/SideGenerator "./"

generator/SideGenerator: generator/SideGenerator.hs generator/HtmlParser.hs generator/CPPParser.hs
	ghc -O3 generator/SideGenerator.hs generator/HtmlParser.hs generator/CPPParser.hs -odir 'generator'

clean:
	rm generator/*.hi
	rm generator/*.o
	rm generator/SideGenerator