/* This is a modified version of original.c that is used to read and convert the data from 
the cafeteria dataset into a dataset of measurements for deep learning/machine learning
programs*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_GESTURES 500
#define MAX_WINDOWS 20000
#define MAX_DATA 54000 /* 1 hr at 15 Hz */

/*Window Calculations*/
#define stride 15 // Multiples of 15 => 1s = 15, 2s = 30 etc.
#define WINDOW_SIZE 30*15 // 30 sec at 15 Hz

int main(int argc, char *argv[])
{
    FILE *fpt;
    float *Data[7];
    float *Conv_Data[7], zero[3], sum;
    int i,j,k,TotalData, TotalGestures = 0, gesture_start[MAX_GESTURES], gesture_end[MAX_GESTURES], gesture_type[MAX_GESTURES];
    int gesture_count[5],a,g,window_start,window_end, flag;
    char c, activity[MAX_GESTURES][11];

    fpt = fopen(argv[1],"r");

    if (fpt == NULL)
    {
        printf("Unable to open bites file, exiting program now! \n");
        exit(0);
    }

    /*Cannot pre-allocate so much data -> stackdump; hence need to calloc it*/
    for (j = 0; j < 7; j++)
    {
        Data[j] = (float *)calloc(MAX_DATA,sizeof(float));
        Conv_Data[j] = (float *)calloc(MAX_DATA,sizeof(float));
    }

    for (j=0; j < 3; j++) /*Initialize zero[3] = 0 before calculation*/
    {
        zero[j] = 0.0;
    }

    i = 0;
    TotalData = 0;
    while(1)
    {
        k = fscanf(fpt, "%f %f %f %f %f %f %f", &Data[0][i], &Data[1][i], &Data[2][i], &Data[3][i], &Data[4][i], &Data[5][i], &Data[6][i]);
        if(k != 7)
        {
            break;
        }
        for (j=0; j < 3; j++)
        {
            zero[j] = zero[j] + Data[j + 3][i];
        }
        TotalData = TotalData + 1;
        i = i + 1;
        if(TotalData >= MAX_DATA)
        {
            printf("Maximum data-size exceeded, increase MAX_DATA \n");
            exit(0);
        }
    }
    fclose(fpt);
    for(j = 0; j < 3; j++)
    {
        zero[j] = zero[j]/(float) TotalData;
    }
        
    for(i=0; i<TotalData; i++)
    {
        for(j=0; j<3; j++)
        {
            Data[j][i] = (Data[j][i] - 1.65)*(5.0/3.3); // Convert accelerometer units from volts to gravities
        }
        for(j=3; j<6; j++)
        {
            Data[j][i] = (Data[j][i] - zero[j-3])*400.0; // Convery gyroscope units from volts to deg/sec
        }        
    }
        
    /*Smooth the data -> 1 sec window, which corresponds to 15 samples or measurements*/
    for(i=0; i<7; i++)
    {
        for(j=0; j<7; j++)
        {
            Conv_Data[j][i] = Data[j][i];
        }
    }

    for(i=TotalData-7; i<TotalData; i++)
    {
        for(j=0; j<7; j++)
        {
            Conv_Data[j][i] = Data[j][i];
        }
    }

    for(i=7; i<TotalData-7; i++)
    {
        for(j=0; j<7; j++)
        {
            sum = 0.0;
            for(k=i-7; k<=i+7; k++)
            {
                if(k >= 0 && k < TotalData)
                {
                    sum = sum + Data[j][k];
                }
            }
            Conv_Data[j][i] = sum/ (float) 15;
        }
    }
    
    /*Read gesture file*/
    fpt = fopen(argv[2],"r");
	if(fpt == NULL)
	{
		printf("Unable to open the gesture file \n");
		exit(0);
	}	    
    i = 0;    
	while(1)
	{
		k = fscanf(fpt,"%s %d %d", activity[i], &gesture_start[i], &gesture_end[i]);
		if (k != 3)
			{
				break;
			}
		
		if (strcmp(activity[i],"bite") == 0)
		{
			gesture_type[i] = 0;
		}
		else if (strcmp(activity[i],"utensiling") == 0)
		{
			gesture_type[i] = 1;
		}
		else if (strcmp(activity[i],"rest") == 0)
		{
			gesture_type[i] = 2;
		}
		else if (strcmp(activity[i],"drink") == 0)
		{
			gesture_type[i] = 3;
		}
		else if (strcmp(activity[i],"other") == 0)
		{
			gesture_type[i] = 4;
		}
		else
		{
			printf("Unknown gesture, exiting loop \n");
			exit(0);
		}

		TotalGestures = TotalGestures + 1;
		i = i + 1;	
	}
	fclose(fpt);    
    
    j = 0;
	window_start = 0;
	window_end = WINDOW_SIZE;

    while(1)
	{
	    for (i = 0; i < 5; i++)
		{
			gesture_count[i] = 0;
	    }

		for(a=window_start; a<window_end; a++)
		{
			flag = 0;
            for(g=0; g<TotalGestures; g++) 
			{
				if(gesture_start[g] <= a && gesture_end[g] >= a)
				{
					// Data index 'a' is inside gesture 'g'
					// gesture_count[gesture_type[g]]++;	
                    printf("%d ",gesture_type[g]);	
                    flag = 1;			
				}              
            }
            if(flag == 0)
            {
                printf("%d ",5); //Marker for unlabeled time-stamp
            }          
            
		}	
	            
	    /*for (i = 0; i < 5; i++)
		{
			printf("%d ",gesture_count[i]);
	    } */ 

        for (i=window_start; i<window_end; i++) 
        {
            printf("%f %f %f %f %f %f ", Conv_Data[0][i], Conv_Data[1][i], Conv_Data[2][i], Conv_Data[3][i], Conv_Data[4][i], Conv_Data[5][i]);
            // Last space is needed for seperation between data entries
        }
        printf("\n");

	    j = j + 1;
	    window_start = window_start + stride;
	    window_end = window_start + WINDOW_SIZE;        

        if(window_end > gesture_end[TotalGestures - 1])
		{
			break;
	    }

	    if( j > MAX_WINDOWS)
	    {
			break;
	    }    
	}    
    
    free(Data[0]);free(Data[1]);free(Data[2]);free(Data[3]);free(Data[4]);free(Data[5]);free(Data[6]);
    free(Conv_Data[0]);free(Conv_Data[1]);free(Conv_Data[2]);free(Conv_Data[3]);free(Conv_Data[4]);free(Conv_Data[5]);free(Conv_Data[6]);

    return 0;
}