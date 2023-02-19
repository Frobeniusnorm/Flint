AHTML := $(wildcard *.ahtml)
all: $(AHTML) generator/SideGenerator
	generator/SideGenerator "./"

generator/SideGenerator: generator/SideGenerator.hs generator/HtmlParser.hs
	ghc generator/SideGenerator.hs generator/HtmlParser.hs -odir 'generator'

