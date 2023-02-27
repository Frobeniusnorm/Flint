AHTML := $(wildcard *.ahtml)
all: $(AHTML) generator/SideGenerator
	generator/SideGenerator "./"

generator/SideGenerator: generator/SideGenerator.hs generator/HtmlParser.hs generator/CPPParser.hs
	ghc generator/SideGenerator.hs generator/HtmlParser.hs generator/CPPParser.hs -odir 'generator'

