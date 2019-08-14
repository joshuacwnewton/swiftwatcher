<img src="data/logo.png" align="right" width="200" height="200"/>

# swiftwatcher
> A collaboration between the University of Victoria's Computer Vision 
Lab and Algoma SwiftWatch

`swiftwatcher` is a open-source tool written in Python which 
automatically counts chimney swifts in video files _(so you don't have 
to!)_ It uses ideas from the field of computer vision to locate and 
track birds within a video's frames, and detect when a bird has entered 
a chimney.

## Getting Started

Swiftwatcher is provided as a single executable file which can be run by
itself. This file can be downloaded from the Releases page, 
[found here](https://github.com/joshuacwnewton/swiftwatcher/releases). 

Each release on the Releases page has an "Assets" section containing 
executable files for both Linux and Windows operating systems. These 
files will contain `win` or `linux` in their names. A demo video is 
also provided.

#### Usage Instructions

Once you have downloaded the file corresponding to your operating system, 
please follow these steps to use swiftwatcher:

1. Place each video you wish to analyse inside a single folder.
2. Run the program you downloaded from the Releases page. 
    * **Linux:** swiftwatcher can be run from the command line by typing

        `./swiftwatcher-linux.0.1.0`
    * **Windows:** swiftwatcher can be run by double-clicking it, or
    from the command line by typing 
    
        `swiftwatcher-win-0.1.0.exe`
3. You will be prompted to select a folder which contains the video
files. Please select the folder created in Step 1.
4. If the folder you selected contains valid video files, you will be
shown a frame from each video, and prompted to select the two corners 
of the chimney in the video. When two points are selected, you can 
proceed with the 'y' key or select again with the 'n' key.
5. When the chimney has been identified for all videos, 
the application will begin to process them to look for swifts.

This process can be viewed in the demonstration below.

<p align="center"><img src="data/screenshots/demo.gif"></p>

Results will be outputted to a folder of the same name as the video 
file. Within this folder, you will find `.csv` files in a number of 
different time formats. Each `.csv` file contains three columns: 

* **TMSTAMP:** The in-video timestamp corresponding to the counted 
swifts.
* **PREDICTED:** What the application counts as a swift entering the 
chimney.
* **REJECTED:** A potential event that did not meet the 
necessary criteria to be counted. 

Counts are only determined from **PREDICTED** swifts, but **REJECTED**
counts are included for transparency.


<p align="center"><img src="data/screenshots/results.png"></p>

## Links

An in-progress research paper will linked shortly. For more information
about the theoretical ideas behind this approach, be sure to check back
soon.

## License

This project is licensed under the GNU General Public License v3.0. To
view the text version of this license, please refer to the `LICENSE`
file included in this project's repository.

