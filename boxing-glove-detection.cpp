/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "detectNet.h"

#include <iostream>
#include <fstream>
#include <array>
#include <ctime>
#include <cmath>
using namespace std;

//#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)	
#define DEFAULT_CAMERA 0

#define TARGET_COUNT 6

#define MODE_CALB 1
#define MODE_PLAY 2

enum class TargetMatchType {
	Calibrate,
	MinRegion, 	// 'cake' region
	MinMaxRegion 	// 'donut' region
};

struct point {
	float x;
	float y;
};

struct target {
	point pt1;
	point pt2;
};

int gameMode = 0;
target targets[6];
const char* targetNames[6] = {"JAB", "CROSS", "LEFT HOOK", "RIGHT HOOK", "LEFT UPPERCUT", "RIGHT UPPERCUT"};
array<int, 8> punchCombos = {0, 1, 0, 0, 1, 0, 1, 2};
const string gameSettingsFileName = "game.settings.txt";

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

bool isTargetMatch(TargetMatchType type, target tgt, float* actual, float threshold)
{
	float pt1xDiff = abs(tgt.pt1.x - actual[0]);
	float pt1yDiff = abs(tgt.pt1.y - actual[1]);
	float pt2xDiff = abs(tgt.pt2.x - actual[2]);
	float pt2yDiff = abs(tgt.pt2.y - actual[3]);

	if (type == TargetMatchType::Calibrate)
	{
		return !(pt1xDiff > threshold && pt1yDiff > threshold && pt2xDiff > threshold && pt2yDiff > threshold);
	}
	else if (type == TargetMatchType::MinMaxRegion)
	{
		return pt1xDiff < threshold && pt1yDiff < threshold && pt2xDiff < threshold && pt2yDiff < threshold;
	}
	else if (type == TargetMatchType::MinRegion)
	{
		bool isPt1Within = actual[0] > (tgt.pt1.x - threshold) && actual[1] > (tgt.pt1.y - threshold);
		bool isPt2Within = actual[2] < (tgt.pt2.x + threshold) && actual[3] < (tgt.pt2.y + threshold);
		return isPt1Within && isPt2Within;
	}
	else
	{
		return false;
	}
}

int main( int argc, char** argv )
{
	printf("detectnet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\ndetectnet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create detectNet
	 */
	detectNet* net = detectNet::Create(argc, argv);
	
	if( !net )
	{
		printf("detectnet-camera:   failed to initialize imageNet\n");
		return 0;
	}

	/*
	 * allocate memory for output bounding boxes and class confidence
	 */
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;

	int targetBoxNum = 1;
	float* targetBoxCPU = NULL;
	float* targetBoxCUDA = NULL;
	float* targetBoxColorCPU = NULL;
	float* targetBoxColorCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) ||
	    !cudaAllocMapped((void**)&targetBoxCPU, (void**)&targetBoxCUDA, targetBoxNum * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&targetBoxColorCPU, (void**)&targetBoxColorCUDA, targetBoxNum * sizeof(float4)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}

	/*
	 * set target box color
	 */
	targetBoxColorCPU[0] = 250.0f;
	targetBoxColorCPU[1] = 50.0f;
	targetBoxColorCPU[2] = 50.0f;
	targetBoxColorCPU[3] = 100.0f;
		
	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\ndetectnet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("detectnet-camera:  failed to create openGL texture\n");
	}
	
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\ndetectnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  camera open for streaming\n");
	
	/*
	 * load game settings
	 */
	int targetIndex = 0;
	int punchComboIndex = 0;
	int calibNoDetectCnt = 0;
	int calibNoDetectMax = 5;
	double fastestSeshTime = 0;
	clock_t calibStartTime = 0;
	clock_t hitTargetStartTime = 0;
	clock_t targetSeshStartTime = 0;
	target previousCalibTarget;
	int calibTargetMaxTimeSec = 3;
	double objConfThreshold = 0.6;
	char gameSettingsFileInput[256];
	float playTargetDiffThreshold = 2;
	float calibTargetDiffThreshold = 15;
	fstream gameSettingsFile (gameSettingsFileName, fstream::in);
	if (gameSettingsFile.is_open())
	{
		string line;
		getline(gameSettingsFile, line);
		gameSettingsFile.close();

		int coordItemIndex = 0;
		char* item = strtok(const_cast<char*>(line.c_str()), ",");
		while (item != NULL)
		{
			float pt = stof(item);
			if (coordItemIndex == 0)
			{
				targets[targetIndex].pt1.x = pt;
				coordItemIndex++;
			}
			else if (coordItemIndex == 1)
			{
				targets[targetIndex].pt1.y = pt;
				coordItemIndex++;
			}
			else if (coordItemIndex == 2)
			{
				targets[targetIndex].pt2.x = pt;
				coordItemIndex++;
			}
			else if (coordItemIndex == 3)
			{
				targets[targetIndex].pt2.y = pt;
				
				coordItemIndex = 0;
				printf("speed-reflex-game: target coord %s retrieved: (%.1f,%.1f), (%.1f,%.1f)\n", targetNames[targetIndex], 
						targets[targetIndex].pt1.x, targets[targetIndex].pt1.y, 
						targets[targetIndex].pt2.x, targets[targetIndex].pt2.y);

				if (++targetIndex >= TARGET_COUNT)
					break;
			}
			else
			{
				printf("speed-reflex-game: error retrieving game settings, entering calibration mode\n");
				gameMode = MODE_CALB;
				break;
			}

			item = strtok(NULL, ",");
		}

		if (targetIndex >= TARGET_COUNT)
		{
			printf("speed-reflex-game: retrieved all game settings, entering play mode\n");
			gameMode = MODE_PLAY;
		}
		else
		{
			printf("speed-reflex-game: unable to retrieve all game settings, entering calibration mode\n");
			gameMode = MODE_CALB;
		}
	}
	else
	{
		printf("speed-reflex-game: unable to read game settings file, entering calibration mode\n");
		gameMode = MODE_CALB;
	}

	if (gameMode == MODE_CALB)
	{
		if (targetIndex != 0) targetIndex = 0;
		previousCalibTarget.pt1.x = 0;
		previousCalibTarget.pt1.y = 0;
		previousCalibTarget.pt2.x = 0;
		previousCalibTarget.pt2.y = 0;
	}
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\ndetectnet-camera:  failed to capture frame\n");

		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
			printf("detectnet-camera:  failed to convert from NV12 to RGBA\n");

		if (gameMode == MODE_CALB)
		{
			char targetNameStr[256];
			sprintf(targetNameStr, "CALIBRATE: %s", targetNames[targetIndex]);
			if (!font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(), targetNameStr, 10, 10))
				printf("speed-reflex-game: failed to draw target name text\n");
			
			CUDA(cudaDeviceSynchronize());

			int elapsedTime = 0;
			if (calibStartTime > 0)
			{
				elapsedTime = 1 + int ((clock() - calibStartTime) / CLOCKS_PER_SEC);

				char elapsedStr[64];
				sprintf(elapsedStr, "HOLD FOR %is (%is)", calibTargetMaxTimeSec, elapsedTime > calibTargetMaxTimeSec ? calibTargetMaxTimeSec : elapsedTime);
				if (!font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(), elapsedStr, 10, 40))
					printf("speed-reflex-game: failed to draw text\n");

				CUDA(cudaDeviceSynchronize());
			}

			int classIndex = 0;
			int numBoundingBoxes = 1;
			if (net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU) 
					&& numBoundingBoxes > 0 // enter only if object was detected
					&& confCPU[classIndex] > objConfThreshold) // enter only if detection above defined threshold
			{
				if (calibNoDetectCnt > 0)
					calibNoDetectCnt = 0;
				
				if (calibStartTime == 0)
					calibStartTime = clock();
				
				if (!net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), 
							bbCUDA, numBoundingBoxes, classIndex))
					printf("speed-reflex-game: failed to draw calibration box\n");

				CUDA(cudaDeviceSynchronize());

				if (previousCalibTarget.pt1.x != 0 && previousCalibTarget.pt1.y != 0 
						&& previousCalibTarget.pt2.x != 0 && previousCalibTarget.pt2.y != 0)
				{
					if (!isTargetMatch(TargetMatchType::Calibrate, previousCalibTarget, bbCPU, calibTargetDiffThreshold))
					{
						calibStartTime = 0;
					}
					else if (elapsedTime > calibTargetMaxTimeSec)
					{
						targets[targetIndex].pt1.x = bbCPU[0];
						targets[targetIndex].pt1.y = bbCPU[1];
						targets[targetIndex].pt2.x = bbCPU[2];
						targets[targetIndex].pt2.y = bbCPU[3];

						// write target coords to string for file output
						if (targetIndex == 0)
						{
							sprintf(gameSettingsFileInput, "%.1f,%.1f,%.1f,%.1f", 
									targets[targetIndex].pt1.x, targets[targetIndex].pt1.y, 
									targets[targetIndex].pt2.x, targets[targetIndex].pt2.y);
						}
						else
						{
							sprintf(gameSettingsFileInput, "%s,%.1f,%.1f,%.1f,%.1f", const_cast<char*>(gameSettingsFileInput), 
									targets[targetIndex].pt1.x, targets[targetIndex].pt1.y, 
									targets[targetIndex].pt2.x, targets[targetIndex].pt2.y);
						}

						// write target coords to file and switch game mode
						if (++targetIndex >= TARGET_COUNT)
						{
							printf("speed-reflex-game: writing game settings to file\n");

							gameSettingsFile.open(gameSettingsFileName, fstream::out);
							gameSettingsFile << const_cast<char*>(gameSettingsFileInput);
							gameSettingsFile.close();

							gameMode = MODE_PLAY;
						}

						calibStartTime = 0;
					}
				}

				previousCalibTarget.pt1.x = bbCPU[0];
				previousCalibTarget.pt1.y = bbCPU[1];
				previousCalibTarget.pt2.x = bbCPU[2];
				previousCalibTarget.pt2.y = bbCPU[3];
			}
			else if (++calibNoDetectCnt >= calibNoDetectMax)
			{
				// reset timer if no detection after defined misses
				calibStartTime = 0;
				calibNoDetectCnt = 0;

				previousCalibTarget.pt1.x = 0;
				previousCalibTarget.pt1.y = 0;
				previousCalibTarget.pt2.x = 0;
				previousCalibTarget.pt2.y = 0;
			}
		}
		else if (gameMode == MODE_PLAY)
		{
			if (hitTargetStartTime == 0)
				hitTargetStartTime = clock();

			if (targetSeshStartTime == 0)
				targetSeshStartTime = clock();

			// classify image with detectNet
			int numBoundingBoxes = maxBoxes;
		
			if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
			{
				//printf("%i bounding boxes detected\n", numBoundingBoxes);
			
				int lastClass = 0;
				int lastStart = 0;
				
				for( int n=0; n < numBoundingBoxes; n++ )
				{
					const int nc = confCPU[n*2+1];
					//float* bb = bbCPU + (n * 4);
					
					//printf("detected obj %i  class #%u (%s)  confidence=%f\n", n, nc, net->GetClassDesc(nc), confCPU[n*2]);
					//printf("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
				
					if( nc != lastClass || n == (numBoundingBoxes - 1) )
					{
						if( !net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), 
							                        bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
							printf("detectnet-console:  failed to draw boxes\n");
						
						lastClass = nc;
						lastStart = n;
	
						CUDA(cudaDeviceSynchronize());
					}
				}
			
				if( display != NULL )
				{
					char str[256];
					sprintf(str, "TensorRT %i.%i.%i | %s | %04.1f FPS", 
							NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, 
							precisionTypeToStr(net->GetPrecision()), display->GetFPS());

					display->SetTitle(str);	
				}	
			}
	
			targetBoxCPU[0] = targets[punchCombos[punchComboIndex]].pt1.x;
			targetBoxCPU[1] = targets[punchCombos[punchComboIndex]].pt1.y;
			targetBoxCPU[2] = targets[punchCombos[punchComboIndex]].pt2.x;
			targetBoxCPU[3] = targets[punchCombos[punchComboIndex]].pt2.y;

			if( !net->DrawTargetBox((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), targetBoxCUDA, targetBoxColorCUDA, 1) )
				printf("speed-reflex-game: failed to draw target box\n");

			CUDA(cudaDeviceSynchronize());
			
			char targetNameStr[256];
			strcpy(targetNameStr, targetNames[punchCombos[punchComboIndex]]);
			if (!font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(), targetNameStr, 10, 10))
				printf("speed-reflex-game: failed to draw target name text\n");
			
			CUDA(cudaDeviceSynchronize());

			if (fastestSeshTime > 0)
			{
				char fastestSeshTimeStr[256];
				sprintf(fastestSeshTimeStr, "FASTEST SESSION: %.2fs", fastestSeshTime);
				if (!font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(), fastestSeshTimeStr, 10, 30))
					printf("speed-reflex-game: failed to draw fastest session text\n");
			}

			if (isTargetMatch(TargetMatchType::MinRegion, targets[punchCombos[punchComboIndex]], bbCPU, playTargetDiffThreshold))
			{
				printf("speed-reflex-game: time to hit %s: %.2fs\n", targetNames[punchCombos[punchComboIndex]], 
						 (clock() - hitTargetStartTime)*1.0 / CLOCKS_PER_SEC);

				hitTargetStartTime = 0;

				if (++punchComboIndex >= punchCombos.size())
				{
					double totalSeshTime = (clock() - targetSeshStartTime)*1.0 / CLOCKS_PER_SEC;
					if (fastestSeshTime == 0 || totalSeshTime < fastestSeshTime)
						fastestSeshTime = totalSeshTime;

					printf("speed-reflex-game: total session time: %.2fs\n", totalSeshTime);

					targetSeshStartTime = 0;

					punchComboIndex = 0;
				}
			}
		}

		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
		 						   camera->GetWidth(), camera->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}
	
	printf("\ndetectnet-camera:  un-initializing video device\n");
	
	
	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("detectnet-camera:  video device has been un-initialized.\n");
	printf("detectnet-camera:  this concludes the test of the video device.\n");
	return 0;
}

