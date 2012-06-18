//---------------------------------------------------------------------------
// Includes
//---------------------------------------------------------------------------
#define SOCNET_CAMERA // use camera
//#define SOCNET_BACKDROP // use backdrop.jpg tracker instead of camera
#define SOCNET_TRACKER // Wrap 920AR tracker 
#define SOCNET_TRACKER_USE // use the tracker

#include "SocNetRender.h"
#include "Geometry.cpp"
#include "Smooth.cpp"
#include "Utils.h"
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#ifdef SOCNET_CAMERA
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#endif
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <string>
#include <map>
#include <vector>
#include <time.h>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <ni/XnOpenNI.h>
#include <ni/XnCppWrapper.h>
#include <boost/thread.hpp>
using namespace std;

extern xn::Context g_Context;
extern xn::UserGenerator g_UserGenerator;
extern xn::DepthGenerator g_DepthGenerator;

extern time_t clickTimer;

extern int testNum;
extern int quitRequested;
extern bool drawSkeleton;
extern bool drawSquare;
extern bool headView;
extern bool doClear;
extern XnUInt32 currentUser;
extern bool isMouseDown;
extern bool isUsingMouse;
extern int currentBrush;
extern int brushCount;
extern float rr;
extern float gg;
extern float bb;
extern float aa;

GLUquadricObj* quadric;

#ifdef SOCNET_CAMERA
GLuint makeTexture(int width, int height, int channels, const GLvoid *data) {
    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, texID);

    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glDisable(GL_TEXTURE_RECTANGLE_ARB);

    if(channels == 3)
        glTexImage2D(GL_TEXTURE_RECTANGLE_ARB,0,GL_RGB, width,height, 0,GL_BGR,GL_UNSIGNED_BYTE, data);
    else if(channels == 4)
        glTexImage2D(GL_TEXTURE_RECTANGLE_ARB,0,GL_RGBA, width,height, 0,GL_BGRA,GL_UNSIGNED_BYTE, data);
        
    return texID;
}
cv::Mat makeTexture(string fileName, GLuint &texID) {
    cv::Mat mat = cv::imread(fileName);
    texID = makeTexture(mat.size().width, mat.size().height, mat.channels(), mat.data);
    return mat;
}

cv::VideoCapture capture;
cv::Mat frame;
cv::Size size;
GLuint cameraTextureID;
boost::thread camThread;
bool cameraInit = false;
bool cameraLock = false;
void initCamera() {
    #ifdef SOCNET_BACKDROP
    frame = cv::imread("../backdrop.jpg");
    #else
    capture.open(-1);
    capture >> frame;
    #endif
    if(frame.data) {
        size = frame.size();        
        cameraTextureID = makeTexture(size.width, size.height, frame.channels(), frame.data);
        cameraInit = true;
    }
}
void readCamera() {
    if(cameraInit==true) {
        #ifdef SOCNET_BACKDROP
        #else
        capture >> frame;
        #endif
    }
    cameraLock = false;
}
void updateCamera() {
    #ifdef SOCNET_BACKDROP
    #else
    if(!cameraLock) {
        cameraLock = true;
        if(frame.data) {
            glBindTexture(GL_TEXTURE_RECTANGLE_ARB, cameraTextureID);
            if(frame.channels() == 3)
                glTexImage2D(GL_TEXTURE_RECTANGLE_ARB,0,GL_RGB, size.width,size.height, 0,GL_BGR,GL_UNSIGNED_BYTE, frame.data);
            else if(frame.channels() == 4)
                glTexImage2D(GL_TEXTURE_RECTANGLE_ARB,0,GL_RGBA, size.width,size.height, 0,GL_BGRA,GL_UNSIGNED_BYTE, frame.data);
        }
        camThread.join();
        camThread = boost::thread(readCamera);
    }
    #endif
}
void renderCamera() {
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, size.width, size.height, 0, -1.0, 1.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, cameraTextureID);
    
    glScalef(1.35,1.35,1);
    glTranslatef(-60,-105,0);
    
    glColor4f(1,1,1,1);
    glBegin(GL_QUADS);
        glTexCoord2i(0,size.height);            glVertex3f(0,size.height, 0);
        glTexCoord2i(size.width,size.height);   glVertex3f(size.width,size.height, 0);
        glTexCoord2i(size.width,0);             glVertex3f(size.width,0, 0);
        glTexCoord2i(0,0);                      glVertex3f(0,0, 0);
    glEnd();
    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}
#endif

#ifdef SOCNET_TRACKER
float initPitch, initYaw, initRoll;
SmoothData *a, *b, *c;

char* trackerBlock = new char[42];
int16_t* trackerBlock16 = (int16_t*)trackerBlock;
double* normalBlock = new double[12];

double HiGyroMulti = (1/4.333333333333)*(M_PI/180.0);

#define GRAVITY 256

double multi[12] = {
    3.7037037037037037/1000, 3.3003300330033003/1000, 3.389830508474576/1000, 
    1*GRAVITY,1*GRAVITY,1*GRAVITY, 
    1,1,1, 
    HiGyroMulti,HiGyroMulti,HiGyroMulti
};
double offsets[12] = {
    100.0, -363.0, -5.0, 
    0,-40,0, 
    -31,22,-16, 
    -87,+40,-30
};

//WE SWAP SOME AXIS RELATIONSHIPS HERE
//TO GET IN LINE WITH AHRS CODE
// negative acc x and y
// negative gyro x then swap x and z
double getMagX() { return normalBlock[0]; }
double getMagY() { return normalBlock[1]; }
double getMagZ() { return normalBlock[2]; }

double getAccX() { return -normalBlock[3]; }
double getAccY() { return -normalBlock[4]; }
double getAccZ() { return normalBlock[5]; }

double getLoGyroX() { return -normalBlock[8]; }
double getLoGyroY() { return normalBlock[7]; }
double getLoGyroZ() { return -normalBlock[6]; }

double getHiGyroX() { return -normalBlock[11]; }
double getHiGyroY() { return normalBlock[10]; }
double getHiGyroZ() { return -normalBlock[9]; }

bool trackerInit = false;
bool trackerLock = false;
boost::thread trackerThread;
ifstream tracker;


void dt(timespec start, timespec end, timespec* dif) {
	if ((end.tv_nsec-start.tv_nsec)<0) {
		dif->tv_sec = end.tv_sec-start.tv_sec-1;
		dif->tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		dif->tv_sec = end.tv_sec-start.tv_sec;
		dif->tv_nsec = end.tv_nsec-start.tv_nsec;
	}
}

////// vector math

//Computes the dot product of two vectors
double Vector_Dot_Product(double vector1[3],double vector2[3])
{
  double op=0;
  
  for(int c=0; c<3; c++)
  {
  op+=vector1[c]*vector2[c];
  }
  
  return op; 
}

//Computes the cross product of two vectors
void Vector_Cross_Product(double vectorOut[3], double v1[3],double v2[3])
{
  vectorOut[0]= (v1[1]*v2[2]) - (v1[2]*v2[1]);
  vectorOut[1]= (v1[2]*v2[0]) - (v1[0]*v2[2]);
  vectorOut[2]= (v1[0]*v2[1]) - (v1[1]*v2[0]);
}

//Multiply the vector by a scalar. 
void Vector_Scale(double vectorOut[3],double vectorIn[3], double scale2)
{
  for(int c=0; c<3; c++)
  {
   vectorOut[c]=vectorIn[c]*scale2; 
  }
}

void Vector_Add(double vectorOut[3],double vectorIn1[3], double vectorIn2[3])
{
  for(int c=0; c<3; c++)
  {
     vectorOut[c]=vectorIn1[c]+vectorIn2[c];
  }
}

//Multiply two 3x3 matrixs. This function developed by Jordi can be easily adapted to multiple n*n matrix's. (Pero me da flojera!). 
void Matrix_Multiply(double a[3][3], double b[3][3],double mat[3][3])
{
  double op[3]; 
  for(int x=0; x<3; x++)
  {
    for(int y=0; y<3; y++)
    {
      for(int w=0; w<3; w++)
      {
       op[w]=a[x][w]*b[w][y];
      } 
      mat[x][y]=0;
      mat[x][y]=op[0]+op[1]+op[2];
      
      //double test=mat[x][y];
    }
  }
}




/////

double G_Dt=0;

// Euler angles
double roll=0;
double pitch=0;
double yaw=0;

double MAG_Heading;

double Accel_Vector[3]= {0,0,0}; //Store the acceleration in a vector
double Gyro_Vector[3]= {0,0,0};//Store the gyros turn rate in a vector
double Omega_Vector[3]= {0,0,0}; //Corrected Gyro_Vector data
double Omega_P[3]= {0,0,0};//Omega Proportional correction
double Omega_I[3]= {0,0,0};//Omega Integrator
double Omega[3]= {0,0,0};

double errorRollPitch[3]= {0,0,0};

double errorYaw[3]= {0,0,0};

double DCM_Matrix[3][3]= {
  {
    1,0,0  }
  ,{
    0,1,0  }
  ,{	
    0,0,1  }
}; 
double Update_Matrix[3][3]={{0,1,2},{3,4,5},{6,7,8}}; //Gyros here


double Temporary_Matrix[3][3]={
  {
    0,0,0  }
  ,{
    0,0,0  }
  ,{
    0,0,0  }
};

double GL_Matrix[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
double GL_MatrixT[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

void Compass_Heading() {
  double MAG_X;
  double MAG_Y;
  double cos_roll;
  double sin_roll;
  double cos_pitch;
  double sin_pitch;
  
  cos_roll = cos(roll);
  sin_roll = sin(roll);
  cos_pitch = cos(pitch);
  sin_pitch = sin(pitch);
  
  // adjust for LSM303 compass axis offsets/sensitivity differences by scaling to +/-0.5 range
  /*c_magnetom_x = (float)(magnetom_x - SENSOR_SIGN[6]*M_X_MIN) / (M_X_MAX - M_X_MIN) - SENSOR_SIGN[6]*0.5;
  c_magnetom_y = (float)(magnetom_y - SENSOR_SIGN[7]*M_Y_MIN) / (M_Y_MAX - M_Y_MIN) - SENSOR_SIGN[7]*0.5;
  c_magnetom_z = (float)(magnetom_z - SENSOR_SIGN[8]*M_Z_MIN) / (M_Z_MAX - M_Z_MIN) - SENSOR_SIGN[8]*0.5;*/
  
  // Tilt compensated Magnetic filed X:
  MAG_X = getMagX()*cos_pitch+getMagY()*sin_roll*sin_pitch+getMagZ()*cos_roll*sin_pitch;
  // Tilt compensated Magnetic filed Y:
  MAG_Y = getMagY()*cos_roll-getMagZ()*sin_roll;
  // Magnetic Heading
  MAG_Heading = atan2(-MAG_Y,MAG_X);
  
  //if(MAG_Heading == nan) quitRequested = true; 
}

void Normalize(void)
{
  double error=0;
  double temporary[3][3];
  double renorm=0;
  
  error= -Vector_Dot_Product(&DCM_Matrix[0][0],&DCM_Matrix[1][0])*.5; //eq.19

  Vector_Scale(&temporary[0][0], &DCM_Matrix[1][0], error); //eq.19
  Vector_Scale(&temporary[1][0], &DCM_Matrix[0][0], error); //eq.19
  
  Vector_Add(&temporary[0][0], &temporary[0][0], &DCM_Matrix[0][0]);//eq.19
  Vector_Add(&temporary[1][0], &temporary[1][0], &DCM_Matrix[1][0]);//eq.19
  
  Vector_Cross_Product(&temporary[2][0],&temporary[0][0],&temporary[1][0]); // c= a x b //eq.20
  
  renorm= .5 *(3 - Vector_Dot_Product(&temporary[0][0],&temporary[0][0])); //eq.21
  Vector_Scale(&DCM_Matrix[0][0], &temporary[0][0], renorm);
  
  renorm= .5 *(3 - Vector_Dot_Product(&temporary[1][0],&temporary[1][0])); //eq.21
  Vector_Scale(&DCM_Matrix[1][0], &temporary[1][0], renorm);
  
  renorm= .5 *(3 - Vector_Dot_Product(&temporary[2][0],&temporary[2][0])); //eq.21
  Vector_Scale(&DCM_Matrix[2][0], &temporary[2][0], renorm);
}

/**************************************************/

double constrain(double x, double a, double b){
    double res = x;
    res = max(a,res);
    res = min(b,res);
    
    return res;
}

#define Kp_ROLLPITCH 0.02
#define Ki_ROLLPITCH 0.00002
//#define Kp_YAW 1.2
#define Kp_YAW 3.5
#define Ki_YAW 0.00002

void Drift_correction(void)
{
  double mag_heading_x;
  double mag_heading_y;
  double errorCourse;
  //Compensation the Roll, Pitch and Yaw drift. 
  static double Scaled_Omega_P[3];
  static double Scaled_Omega_I[3];
  double Accel_magnitude;
  double Accel_weight;
  
  
  //*****Roll and Pitch***************

  // Calculate the magnitude of the accelerometer vector TODOxxx
  //Accel_magnitude = sqrt(Accel_Vector[0]*Accel_Vector[0] + Accel_Vector[1]*Accel_Vector[1] + Accel_Vector[2]*Accel_Vector[2]);
  //Accel_magnitude = Accel_magnitude / GRAVITY; // Scale to gravity.
  Accel_magnitude = sqrt(Accel_Vector[0]*Accel_Vector[0] + Accel_Vector[1]*Accel_Vector[1] + Accel_Vector[2]*Accel_Vector[2]);
  Accel_magnitude = Accel_magnitude / GRAVITY; // Scale to gravity.
    
  // Dynamic weighting of accelerometer info (reliability filter)
  // Weight for accelerometer info (<0.5G = 0.0, 1G = 1.0 , >1.5G = 0.0)
  Accel_weight = constrain(1 - 2*abs(1 - Accel_magnitude),0,1);  //  

  Vector_Cross_Product(&errorRollPitch[0],&Accel_Vector[0],&DCM_Matrix[2][0]); //adjust the ground of reference
  Vector_Scale(&Omega_P[0],&errorRollPitch[0],Kp_ROLLPITCH*Accel_weight);
  
  Vector_Scale(&Scaled_Omega_I[0],&errorRollPitch[0],Ki_ROLLPITCH*Accel_weight);
  Vector_Add(Omega_I,Omega_I,Scaled_Omega_I);     
  
  //*****YAW***************
  // We make the gyro YAW drift correction based on compass magnetic heading
 
  mag_heading_x = cos(MAG_Heading);
  mag_heading_y = sin(MAG_Heading);
  errorCourse=(DCM_Matrix[0][0]*mag_heading_y) - (DCM_Matrix[1][0]*mag_heading_x);  //Calculating YAW error
  Vector_Scale(errorYaw,&DCM_Matrix[2][0],errorCourse); //Applys the yaw correction to the XYZ rotation of the aircraft, depeding the position.
  
  Vector_Scale(&Scaled_Omega_P[0],&errorYaw[0],Kp_YAW);//.01proportional of YAW.
  Vector_Add(Omega_P,Omega_P,Scaled_Omega_P);//Adding  Proportional.
  
  Vector_Scale(&Scaled_Omega_I[0],&errorYaw[0],Ki_YAW);//.00001Integrator
  Vector_Add(Omega_I,Omega_I,Scaled_Omega_I);//adding integrator to the Omega_I
}
/**************************************************/
/*
void Accel_adjust(void)
{
 Accel_Vector[1] += Accel_Scale(speed_3d*Omega[2]);  // Centrifugal force on Acc_y = GPS_speed*GyroZ
 Accel_Vector[2] -= Accel_Scale(speed_3d*Omega[1]);  // Centrifugal force on Acc_z = GPS_speed*GyroY 
}
*/
/**************************************************/

void Matrix_update(void) {
  Gyro_Vector[0]=getHiGyroX(); //gyro x roll
  Gyro_Vector[1]=getHiGyroY(); //gyro y pitch
  Gyro_Vector[2]=getHiGyroZ(); //gyro Z yaw
  
  /*printf("gyro: %f %f %f\n", 
    Gyro_Vector[0]/G_Dt,
      Gyro_Vector[1]/G_Dt,
        Gyro_Vector[2]/G_Dt
  );*/
  
  Accel_Vector[0]=getAccX();
  Accel_Vector[1]=getAccY();
  Accel_Vector[2]=getAccZ();
    
  Vector_Add(&Omega[0], &Gyro_Vector[0], &Omega_I[0]);  //adding proportional term
  Vector_Add(&Omega_Vector[0], &Omega[0], &Omega_P[0]); //adding Integrator term

  Update_Matrix[0][0]=0;
  Update_Matrix[0][1]=-G_Dt*Omega_Vector[2];//-z
  Update_Matrix[0][2]=G_Dt*Omega_Vector[1];//y
  Update_Matrix[1][0]=G_Dt*Omega_Vector[2];//z
  Update_Matrix[1][1]=0;
  Update_Matrix[1][2]=-G_Dt*Omega_Vector[0];//-x
  Update_Matrix[2][0]=-G_Dt*Omega_Vector[1];//-y
  Update_Matrix[2][1]=G_Dt*Omega_Vector[0];//x
  Update_Matrix[2][2]=0;

  Matrix_Multiply(DCM_Matrix,Update_Matrix,Temporary_Matrix); //a*b=c

  for(int x=0; x<3; x++) //Matrix Addition (update)
  {
    for(int y=0; y<3; y++)
    {
      DCM_Matrix[x][y]+=Temporary_Matrix[x][y];
    } 
  }
}

void Euler_angles(void)
{
  pitch = -asin(DCM_Matrix[2][0]);
  roll = atan2(DCM_Matrix[2][1],DCM_Matrix[2][2]);
  yaw = atan2(DCM_Matrix[1][0],DCM_Matrix[0][0]);

//  	printf("!ANG:%f,%f,%f\n",roll,pitch,yaw);	  
}

void GLMatrix(void)
{
 GL_Matrix[0]=DCM_Matrix[0][0];
 GL_Matrix[4]=DCM_Matrix[0][1];
 GL_Matrix[8]=DCM_Matrix[0][2];
 GL_Matrix[12]=0;

 GL_Matrix[1]=DCM_Matrix[1][0];
 GL_Matrix[5]=DCM_Matrix[1][1];
 GL_Matrix[9]=DCM_Matrix[1][2];
 GL_Matrix[13]=0;

 GL_Matrix[2]=DCM_Matrix[2][0];
 GL_Matrix[6]=DCM_Matrix[2][1];
 GL_Matrix[10]=DCM_Matrix[2][2];    
 GL_Matrix[14]=0;

 GL_Matrix[3]=0;
 GL_Matrix[7]=0;
 GL_Matrix[11]=0;
 GL_Matrix[15]=1;

 GL_MatrixT[0]=DCM_Matrix[0][0];
 GL_MatrixT[1]=DCM_Matrix[0][1];
 GL_MatrixT[2]=DCM_Matrix[0][2];
 GL_MatrixT[3]=0;

 GL_MatrixT[4]=DCM_Matrix[1][0];
 GL_MatrixT[5]=DCM_Matrix[1][1];
 GL_MatrixT[6]=DCM_Matrix[1][2];
 GL_MatrixT[7]=0;

 GL_MatrixT[8]=DCM_Matrix[2][0];
 GL_MatrixT[9]=DCM_Matrix[2][1];
 GL_MatrixT[10]=DCM_Matrix[2][2];    
 GL_MatrixT[11]=0;

 GL_MatrixT[12]=0;
 GL_MatrixT[13]=0;
 GL_MatrixT[14]=0;
 GL_MatrixT[15]=1;

}

////////


timespec timer, timer_old, diff;


void updateTracker() {
    if(trackerInit){
        for(int i=0; i<12; i++) {
            normalBlock[i] = (trackerBlock16[i+1]+offsets[i])*multi[i]/1000.0;            
        }
        
        timer_old = timer;
        clock_gettime(CLOCK_REALTIME, &timer);
        dt(timer_old, timer, &diff);
        G_Dt = (diff.tv_sec + (diff.tv_nsec/1000000000.0));   // Real time of loop run. We use this on the DCM algorithm (gyro integration time)
                
        // *** DCM algorithm
        Compass_Heading(); // Calculate magnetic heading  
        
        // Calculations...
        Matrix_update(); 
        Normalize();
        GLMatrix();
        Drift_correction();
        Euler_angles();
                // ***
    }
}
void readTracker() {
    if(trackerInit) {
        while(quitRequested == 0) {
            trackerLock=true;
            tracker.read(trackerBlock, 42);
            updateTracker();
            trackerLock=false;
        }
    }
}

// tracker.read blocks if no data.
boost::thread firstReadThread;
void firstRead() {
    if(tracker.is_open()) {
        tracker.rdbuf()->pubsetbuf(0, 0);
        tracker.read(trackerBlock, 42);
        if(trackerBlock16[0] == -32767) {
            clock_gettime(CLOCK_REALTIME, &timer);
            trackerInit = true;
            fprintf(stderr, "Tracker initialized.\n");
            trackerThread = boost::thread(readTracker);
        } else {
            fprintf(stderr, "Can't read from tracker: %d\n", trackerBlock16[0]);
            quitRequested = 2;
        }
    } else {
        fprintf(stderr, "Can't open tracker.\n");
        quitRequested = 2;
    }
}

extern char* trackerDevice;
void initTracker() {
    if(!trackerInit) {
        fprintf(stderr, "Tracker init started.\n");
        tracker.open(trackerDevice, ios::in|ios::binary);
        #ifdef SOCNET_TRACKER_USE 
        firstReadThread = boost::thread(firstRead);
        #endif
    }
}
#endif

SmoothPoint *headpos;
SmoothPoint *RightHand, *RightHandProj, *RightElbow, *RightElbowProj;
SmoothPoint *LeftHand, *LeftHandProj, *LeftElbow, *LeftElbowProj;
extern float currentThickness;

Line currentLine;
Lines lines;

bool menuIsSelected = false;
void prevBrush() {
    currentBrush -= 1; 
    if(currentBrush<0) currentBrush = brushCount-1;
}
void nextBrush() {
    currentBrush += 1;
    currentBrush %= brushCount;
}

bool firstRender = true;
bool firstUser = true;

void glPrintString(void *font, char *str) {
    int i,l = strlen(str);

    for(i=0; i<l; i++) glutBitmapCharacter(font,*str++);
}

XnPoint3D GetLimbPosition(XnUserID player, XnSkeletonJoint eJoint) {
    XnSkeletonJointPosition joint;
    g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(player, eJoint, joint);
    
    return joint.position;
}
XnPoint3D getProj(XnPoint3D current) {
    return current;
    /*XnPoint3D proj = xnCreatePoint3D(3,3,3);
    g_DepthGenerator.ConvertRealWorldToProjective(1, &current, &proj);
    return proj;*/
}

void DrawLimb(XnUserID player, XnSkeletonJoint eJoint1, XnSkeletonJoint eJoint2) {
    if(!g_UserGenerator.GetSkeletonCap().IsTracking(player)) return;
    
    XnSkeletonJointPosition joint1, joint2;
    g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(player, eJoint1, joint1);
    g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(player, eJoint2, joint2);
    
    if(joint1.fConfidence <= 0 || joint2.fConfidence <= 0) return;
    
    XnPoint3D pt[2] = {joint1.position, joint2.position};
    
    g_DepthGenerator.ConvertRealWorldToProjective(2, pt, pt);

    glVertex3f(pt[0].X, pt[0].Y, pt[0].Z);
    glVertex3f(pt[1].X, pt[1].Y, pt[1].Z);
}
void DrawLine2(XnPoint3D p1, XnPoint3D p2, float th) {
    glDisable(GL_LIGHTING);
    glBegin(GL_QUADS);
    glVertex3f(p1.X-th, p1.Y, p1.Z);
    glVertex3f(p1.X+th, p1.Y, p1.Z);
    glVertex3f(p2.X+th, p2.Y, p2.Z);
    glVertex3f(p2.X-th, p2.Y, p2.Z);
    glEnd();
    glEnable(GL_LIGHTING);
}
void DrawLine(XnPoint3D p1, XnPoint3D p2) {
    glColor4f(0.9,0.75,0.75,1);
    float th = 2;
    DrawLine2(p1,p2,th);
}


void drawQuad(float x, float y) {
    glBegin(GL_QUADS);
        glVertex3f(-x,-y, 0);
        glVertex3f(-x,+y, 0);
        glVertex3f(+x,+y, 0);
        glVertex3f(+x,-y, 0);
    glEnd();    
}
void drawQuad(float x1, float y1, float x2, float y2) {
    glBegin(GL_QUADS);
        glVertex3f(x1,y1, 0);
        glVertex3f(x1,y2, 0);
        glVertex3f(x2,y2, 0);
        glVertex3f(x2,y1, 0);
    glEnd();
}
void drawTexQuad(float x, float y, int size, GLuint texID) {
    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, texID);
        
    glBegin(GL_QUADS);
    glTexCoord2i(0,size);    glVertex3f(-x,+y, 0);
    glTexCoord2i(size,size); glVertex3f(+x,+y, 0);
    glTexCoord2i(size,0);    glVertex3f(+x,-y, 0);
    glTexCoord2i(0,0);       glVertex3f(-x,-y, 0);
    glEnd();

    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glDisable(GL_TEXTURE_2D);
}

void cleanupSocNet() {
    if(quitRequested==0) quitRequested = 1;
    #ifdef SOCNET_TRACKER
        fprintf(stderr, "Releasing tracker.\n");
        #ifdef SOCNET_TRACKER_USE
        trackerThread.join();
        #endif
        tracker.close();
    #endif
    #ifdef SOCNET_CAMERA
        fprintf(stderr, "Releasing camera.\n");
        camThread.join();
        capture.release();
    #endif
}

XnPoint3D convertKinect(XnPoint3D in) {
    XnPoint3D out = { -in.Z, in.X, -in.Y };
    return out;
}
XnPoint3D convertGlasses(XnPoint3D in) {
    XnPoint3D out = { in.Y, -in.Z, -in.X };
    return out;
}
float distance(XnPoint3D p1, XnPoint3D p2) {
    return
        abs(p1.X-p2.X) +
        abs(p1.Y-p2.Y) +
        abs(p1.Z-p2.Z);
}

bool hadKinect = false;


struct Conn {
    int src;
    int dst;
};
struct Color {
    float R;
    float G;
    float B;
};
struct Node {
    XnPoint3D pos;
    float size;
    Color color;
};

const int NODES = 400;
const int CONNS = 200;
Node nodes [NODES];
Conn conns [CONNS];
int nodeI=-1;//selected node

bool graphGenerated = false;
boost::thread graphThread;
XnPoint3D graphCenter;
void generateGraph() {
    for(int i=0; i<NODES; i++) {
        float range=1000;
        float bestDist=0;
        XnPoint3D bestDistP = {1,1,1};
        if(i==0) {
            bestDistP = {
                graphCenter.X,
                graphCenter.Y,
                graphCenter.Z
            };
        } else for(int k=0; k<100; k++) {
            XnPoint3D randP = {
                graphCenter.X+rand01()*range-range/2.0f,
                graphCenter.Y+rand01()*range-range/2.0f,
                graphCenter.Z+rand01()*range-range/2.0f
            };
            printf("%f,  %f,  %f\n", randP.X, randP.Y, randP.Z);
            float dist=100000;
            XnPoint3D distP = {1,1,1};
            for(int j=0; j<i-1; j++) {
                float currDist = distance(randP, nodes[j].pos);
                if(currDist < dist) {
                    dist = currDist;
                    distP = { randP.X, randP.Y, randP.Z };
                }
            }
            if(dist > bestDist) {
                bestDist = dist;
                bestDistP = { distP.X, distP.Y, distP.Z };
            }
        }
        printf("%f,  %f,  %f\n", bestDistP.X, bestDistP.Y, bestDistP.Z);
        nodes[i].pos = { bestDistP.X, bestDistP.Y, bestDistP.Z };
        nodes[i].size = 10+rand01()*10;
        float r,g,b;
        randomColor(r,g,b);
        nodes[i].color = {r,g,b};
    }
    for(int i=0; i<CONNS; i++) {
        int from=rand()%NODES;
        int to=rand()%NODES;
        if(from==to) from = (from+rand()%(NODES-1))%NODES;
        for(int j=0; j<i; j++) {
            if(from==conns[i].src && to==conns[i].dst) {
                j=0;
                from=rand()%NODES;
                to=rand()%NODES;
                if(from==to) from = (from+rand()%(NODES-1))%NODES;
            }
        }
        
        conns[i] = {from,to};
    }
    graphGenerated = true;
}

void drawBall(XnPoint3D pos, float size) {
    glPushMatrix();
    glTranslatef(pos.X, pos.Y, pos.Z);
    gluSphere(quadric, size, 25, 25);
    glPopMatrix();
}
void drawNode(Node n) {
    glColor4f(n.color.R, n.color.G, n.color.B, 0.75);
    drawBall(n.pos, n.size);
}

float maxX=-10000, minX=+10000;
int cnt = 0;
void renderSocNet() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    cnt++;
    if(firstRender) {
        fprintf(stderr, "Render init.\n");
        // init quadric object
        quadric = gluNewQuadric();
        gluQuadricNormals(quadric, GLU_SMOOTH);
        
        glDisable(GL_CULL_FACE);
        // init camera and its texture
        #ifdef SOCNET_CAMERA
        initCamera();
        #endif
        
        fprintf(stderr, "Render init done.\n");
    }
    
    #ifdef SOCNET_CAMERA
    updateCamera();
    renderCamera();
    #endif
   
    //glDisable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(10, (640/480+0.0), 0.1, 4000.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(0,0,0,
              10,0,0, // look-at vector
              0, 0, -1);// up vector 
    
    glMultMatrixd(GL_MatrixT);
    glRotatef(25,0,0,1); // rotation correction
    glRotatef(-8,0,1,0);
    
    
    if(headpos != NULL) glTranslatef(headpos->Z(), -headpos->X(), headpos->Y()+70);

    if(doClear) {
        doClear = false;
        lines.Clear();
    }
    
    g_Context.WaitAnyUpdateAll();

    XnUserID aUsers[15];
    XnUInt16 nUsers = 15;
    g_UserGenerator.GetUsers(aUsers, nUsers);
    
    if(nUsers>0 && currentUser>=0 && currentUser<=nUsers) {
        int i = currentUser-1;
        {
            // oh, wow
            XnPoint3D handRight0 = GetLimbPosition(aUsers[i], XN_SKEL_RIGHT_HAND);
            XnPoint3D elbowRight0 = GetLimbPosition(aUsers[i], XN_SKEL_RIGHT_ELBOW);
            handRight0 = Vec3::makeLonger(elbowRight0, handRight0, -170-27);
            XnPoint3D handRight0proj = getProj(handRight0);
            XnPoint3D handRight = convertKinect(handRight0);
            XnPoint3D handRightproj = convertKinect(handRight0proj);
            XnPoint3D elbowRightproj = convertKinect(getProj(elbowRight0));

            XnPoint3D handLeft0 = GetLimbPosition(aUsers[i], XN_SKEL_LEFT_HAND);
            XnPoint3D elbowLeft0 = GetLimbPosition(aUsers[i], XN_SKEL_LEFT_ELBOW);
            handLeft0 = Vec3::makeLonger(elbowLeft0, handLeft0, -170-27);
            XnPoint3D handLeft0proj = getProj(handLeft0);
            XnPoint3D handLeft = convertKinect(handLeft0);
            XnPoint3D handLeftproj = convertKinect(handLeft0proj);
            XnPoint3D elbowLeftproj = convertKinect(getProj(elbowLeft0));

            if(firstUser) {
                int s = 25; //smoothing
                RightHand = new SmoothPoint(handRight, s, 1);
                RightHandProj = new SmoothPoint(handRightproj, s, 1);
                RightElbowProj = new SmoothPoint(elbowRightproj, s, 1);
                
                LeftHand = new SmoothPoint(handLeft, s, 1);
                LeftHandProj = new SmoothPoint(handLeftproj, s, 1);
                LeftElbowProj = new SmoothPoint(elbowLeftproj, s, 1);
            } else {
                RightHand->insert(handRight);
                RightHandProj->insert(handRightproj);
                RightElbowProj->insert(elbowRightproj);

                LeftHand->insert(handLeft);
                LeftHandProj->insert(handLeftproj);
                LeftElbowProj->insert(elbowLeftproj);
            }
        }

        XnPoint3D head = getProj(GetLimbPosition(aUsers[i], XN_SKEL_HEAD));
        if(firstUser) {
            headpos = new SmoothPoint(head, 50, 0);
            graphCenter = convertKinect(headpos->get());
            graphThread = boost::thread(generateGraph);
        } else {
            headpos->insert(head);
        }
        firstUser = false;

        if(graphGenerated) {
            float minDist=100000000;
            int minDistI=0;
            for(int i=0; i<NODES; i++) {
                float dist = distance(RightHand->get(), nodes[i].pos);
                if(dist < minDist) { minDist = dist; minDistI = i; }
                
                drawNode(nodes[i]);
            }
            if(isMouseDown) {
                if(nodeI==-1) nodeI = minDistI; else minDistI = nodeI;
                int i = nodeI;
                float alp = 0.8;
                nodes[i].pos = {
                    nodes[i].pos.X*alp+RightHand->get().X*(1-alp),
                    nodes[i].pos.Y*alp+RightHand->get().Y*(1-alp),
                    nodes[i].pos.Z*alp+RightHand->get().Z*(1-alp)
                };
            } else {
                nodeI = -1;
            }
            
            for(int i=0; i<CONNS; i++) {
                glColor4f(1,1,1,0.75);
                DrawLine2(nodes[conns[i].src].pos, nodes[conns[i].dst].pos, 1);
            }
        
            glColor4f(
                0.15,//*mixing + shade*(1-mixing),
                0.15,//*mixing + shade*(1-mixing),
                0.15,//*mixing + shade*(1-mixing),
                0.5//visible[i]?1:0.5
            );
            drawBall(nodes[minDistI].pos, nodes[minDistI].size+4.5);
        }
        
        glColor4f(1,0.7,0.7,0.45);
        DrawLine2(RightHandProj->get(), RightElbowProj->get(), 4);
    } else {
        if(hadKinect && time(0)-15 > clickTimer) {
            quitRequested = 1;
        } else if(!hadKinect && time(0)-45 > clickTimer) {
            quitRequested = 1;
        }
    }

    if(firstRender) {
        fprintf(stderr, "First render done.\n");
    }
    firstRender = false;
    glutSwapBuffers();
}
