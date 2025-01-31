// cropAndStraightenBatch.jsx
// Copyright 2006-2008
// Written by Jeffrey Tranberry
// Photoshop for Geeks Version 2.0

/*
Description:
This script demonstates how to batch process
a folder of images using the crop and straighten command
*/

// enable double clicking from the
// Macintosh Finder or the Windows Explorer
#target photoshop

// Make Photoshop the frontmost application
// in case we double clicked the file
app.bringToFront();

/////////////////////////
// SETUP
/////////////////////////

// A list of file extensions to skip, keep them lower case
gFilesToSkip = Array( "db", "xmp", "thm", "txt", "doc", "md0", "tb0", "adobebridgedb", "adobebridgedbt", "bc", "bct" );

/////////////////////////
// MAIN
/////////////////////////

main();

function main() {
	//Make sure there are no open documents
	if (app.documents.length > 0){
		alert ("This script requires that there are no open documents to run.");
	}else{

		// Pops open a dialog for the user to choose the folder of documents to process
		var inputFolder = Folder.selectDialog("Select a folder of documents to process");
		// Pops open a dialog for the user to set the output folder
		var outputFolder = Folder.selectDialog("Select a folder for the output files");

		// Check to make sure outputfolder is not in inputFolder
		if(outputFolderIsOk(inputFolder, outputFolder) === false){
      		alert("Your output folder should not be within your input folder. It will create an infinitive loop.");
			return;
		}

		// Open and process a folder of Images
		OpenFolderRecursively(inputFolder, outputFolder);

	}
}

/////////////////////////
// FUNCTIONS
/////////////////////////

// Given the a Folder of files, open the files and process them
function OpenFolderRecursively(inputFolder, outputFolder) {
        var filesOpened = 0;
        var fileList = inputFolder.getFiles();
        for ( var i = 0; i < fileList.length; i++ ) {
        		   // Make sure all the files in the folder are compatible with PS
                if ( fileList[i] instanceof File && ! fileList[i].hidden && ! IsFileOneOfThese( fileList[i], gFilesToSkip )) {
                        open( fileList[i] );
                        filesOpened++;

					/////////////////////////
					// Put all your processing functions...
					/////////////////////////

							// Create a variable to store a reference to
							// the currently active document, which in this
							// case is the parent document we want to extract
							// multiple scanned images from
							var docRef = app.activeDocument;

							// Run the cropAndStraighten function
							// which will rusult in more than one open document
							try {
									cropAndStraighten();
							} catch (error) {
									// Skip cropping, image incompatible
							}

							// Close the parent document we originally opened
							docRef.close(SaveOptions.DONOTSAVECHANGES);

							// Process all open documents until no documents
							// are left open.
							while (app.documents.length >=1){

								/////////////////////////
								// Put all your processing functions...
								/////////////////////////

									// Flatten the document in case the file type we want to save to requires a flat doc
									app.activeDocument.flatten();

									//Save as a JPEG to the outputFolder
									var jpegOptions = new JPEGSaveOptions();
									jpegOptions.quality = 12;
									jpegOptions.embedColorProfile = false;
									app.activeDocument.saveAs( File( outputFolder  + "/" + activeDocument.name + ".jpg"), jpegOptions, false);

									// Close without saving
									 app.activeDocument.close(SaveOptions.DONOTSAVECHANGES);

								/////////////////////////
								// ...in the area between these two comments.
								/////////////////////////
							}

					/////////////////////////
					// ...in the area between these two comments.
					/////////////////////////

				}

				// go recursive
				if (!(fileList[i] instanceof File)) {
					var newOuputFolder = new Folder(outputFolder + "/" + new Folder(fileList[i]).name);
					newOuputFolder.create();
					OpenFolderRecursively(fileList[i], newOuputFolder);
				}
        }
        return filesOpened;
}

// given a file name and a list of extensions
// determine if this file is in the list of extensions
function IsFileOneOfThese( inFileName, inArrayOfFileExtensions ) {
	var lastDot = inFileName.toString().lastIndexOf( "." );
	if ( lastDot == -1 ) {
		return false;
	}
	var strLength = inFileName.toString().length;
	var extension = inFileName.toString().substr( lastDot + 1, strLength - lastDot );
	extension = extension.toLowerCase();
	for (var i = 0; i < inArrayOfFileExtensions.length; i++ ) {
		if ( extension == inArrayOfFileExtensions[i] ) {
			return true;
		}
	}
	return false;
}

// Crop and Straighten function created
// using the ScriptingListener plug-in
function cropAndStraighten(){
	var id333 = stringIDToTypeID( "CropPhotosAuto0001" );
	executeAction( id333, undefined, DialogModes.NO );
}

// Validate output is outside input folder
function outputFolderIsOk(inputFolder, outputFolder){
	inputFolderString = inputFolder.absoluteURI;
	outputFolderString = outputFolder.absoluteURI;
	if(inputFolderString === outputFolderString){
		return false;
	}
	if(isChildFolderOf(outputFolderString, inputFolderString)){
		return false;
	}

	return true;
}

function isChildFolderOf(outputFolderString, inputFolderString) {
	inputFolders = inputFolderString.split('/');
	outputFolders = outputFolderString.split('/');

	// Impossible to be child
	if(inputFolders.length > outputFolders.length){
		return false;
	}

	lastFolderIndex = inputFolders.length - 1;
	lastInputFolder = inputFolders[lastFolderIndex];

	if(outputFolders[lastFolderIndex] === lastInputFolder){
		return true;
	}

	return false;
}