# What is Pollenyzer

Pollen is known to be the only source of proteins and fats for honey bees. Therefore it is an important part of the honey bees’ nutrition. They are essential for brood care and a good indicator of the quality of a site. It is also known that a diverse diet makes colonies more robust, also in relation to winter losses. In this work, an app is presented that allows to quantify the pollen from a pollen trap and to determine its colour diversity in an automatic way. The colour diversity is closely related to the actual plant diversity. This correlation allows conclusions to be drawn on the quality of a site and on biodiversity in general. In this way, the app provides beekeepers with important information about the well-being of their colonies, while scientists can benefit from aggregated information about local biodiversity. The app is available as a web app on all devices.

# About this repo
This *beesypollen* repo was published together with the paper [ref] and offers the possibility to detect pollen and extract their colours without using the Pollenyzer web app interface. This can be useful to adapt the code to your own needs or to run the pollen detection automatically on multiple images. If neither of these apply, I recommend using the web interface: <https://pollenyzer.github.io>.

In order to use the colour calibration feature of pollen images please take a look at *beesycc* repository here: <https://github.com/pollenyzer/beesycc>

# Installation

This module is installable and once installed can be imported typing `import beesypollen`.

Dev installation prevents pip from copying the files to python's site package directory but instead keeps the source file where they are. This means that all changes to the source file are automatically "updating" the installed package.

Tested on Ubuntu with python 3.7.

1) Go to directory where `setup.py` is placed.
2) Type `pip install -e .`
3) Install all dependencies `pip install -r requirements.txt`
3) Verify that pip is simply referencing the source files of this project by typing `pip list | grep beesypollen`.


# Usage

Have a look at the `tests/` directory. You can use the code there as a starting point. In `tests/test_resources` you can find example image data to work with.
