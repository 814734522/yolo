#include "stdafx.h"
#include <opencv2\dnn.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include "Windows.h"
#include "time.h"
#include <assert.h>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <mysql.h>


using namespace cv;
using namespace std;
using namespace dnn;

// ������ʼ��
float confThreshold = 0.7;						// ���Ŷ���ֵ
float nmsThreshold = 0.7;						// ����ֵ������ֵ
int inpWidth = 416;								// ����ͼƬ���
int inpHeight = 416;							// ����ͼƬ�߶�
int savevideo = 0;								// ������Ƶ����ͼƬ��0Ϊ����ͼƬ�����1Ϊ������Ƶ�����
bool isdetect = true;							// �Ƿ���м��
string cameraId = "1";							// ��������

//ȫ�ֱ���
long currentFrame = 1;							//��ǰ֡��
char imgPath[20];
vector<string> classes;
string sortedName = "";							//����õļ����Ϣ�������˱��-λ�ñ��-�����-ͼƬ�ļ���
string videosavedir = "./vedio/";

string weight_JL = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\yolov3_best.weights";
string model_JL = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\yolov3.cfg";
string class_JL = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\coco.names";

string weight_JL2 = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\1\\yolov3_best.weights";
string model_JL2 = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\1\\yolov3.cfg";
string class_JL2 = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\1\\JL.names";


string weightFiledir = "E:\\data\\test\\yolov3.weights";
string modelFiledir = "E:\\data\\test\\yolov3.cfg";
string classesFiledir = "E:\\data\\test\\coco.names";



void postprocess(Mat& frame, const vector<Mat>& outs);

void drawPred(int classId, float confidence, int left, int top, int right, int bottom, Mat& frame);

vector<String> getOutputsNames(const Net& net);

void detect_image(Mat frame, string modelWeights, string modelConfiguration, string classesFile);

void getInfo(vector<vector<int> > info);

string get_time(bool savevideo);

void sendInfo(string imgDir, string detectResults);



//-------------------------------------------�����-------------------------------------------
void Open_Camera(string user, string pwd, string ip, string port, bool isdetect) 
{
	/*
	������
	string user:����û���
	string pwd:	�û�����
	string ip:	���ip��ַ
	string port �˿ں�
	*/
	//rstp������������
	String dir1 = format
	("rtsp://%s:%s@%s:%s/MPEG-4/ch1/main/av_stream", user, pwd, ip, port);
	cv::VideoCapture cap("E:\\data\\test\\car.mp4");
	VideoWriter outputvideo;
	Mat frame;
	Mat videoframe;
	//�Ƿ񱣴���Ƶ�ļ�
	if (savevideo) {
		//����������ͷ
		cv::VideoCapture Cap0(dir1);
		//������Ƶ·����ʱ��������
		string filename = get_time(savevideo);
		//���ñ�����Ƶ��ʽ���
		cv::Size s = cv::Size((int)cap.get
			(CV_CAP_PROP_FRAME_WIDTH),(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

		outputvideo.open(filename, -1, 25.0, s, true);
		//��֡������Ƶ
		while (true) {
			Cap0 >> videoframe;
			if (videoframe.empty()) break;
			cv::imshow("output", videoframe);
			//д����Ƶ
			outputvideo << videoframe;
			if (char(waitKey(1)) == 'q') break;
		}
	}
	while (isdetect) {
		
		cap >> frame;
		//����ͷ�Ƿ��
		if (frame.empty()) {
			printf("ERROR: cramera is not opened  !!!");
			break;
		}
		if (currentFrame % 15 == 0) {
			//cout << "ERROR: Vapture is closed!" << endl;
			detect_image(frame, weightFiledir, modelFiledir, classesFiledir);
			imshow("window", frame);
		}
		
		currentFrame++;
		waitKey(1);
		if (waitKey() == 'q') {
			break;
		}
		
	}
	cap.release();
	destroyAllWindows();
}


//-------------------------------------------YOLOV3-------------------------------------------

void detect_image(Mat frame, string modelWeights, string modelConfiguration, string classesFile)
{
	/*
	����������
	Mat frame: ͼƬ��Mat����
	string modelWeights: ģ��Ȩ���ļ�·��
	string modelConfiguration: ģ�������ļ�·��
	string classesFile: ����ļ�·��
	*/
	//���м�������ļ��е�����
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	//��������
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);					

	//����ͼƬ/��Ƶ��������ͷ������ļ�·��
	string str, outputFile;
	VideoWriter video;
	//��������
	static const string kWindows = "Deep Learning object detection in OpenCV";
	namedWindow(kWindows, WINDOW_NORMAL);

	Mat blob;
	blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight),Scalar(0, 0, 0), true, false);

	//�����������
	net.setInput(blob);

	//��ǰ�������粢������������
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	//ȥ�����ŶȵĿ�
	postprocess(frame, outs);

	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame :  %.2f ms", t);
	putText(frame, label, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));

	//�����ļ���ʱ������
	string filename = get_time(savevideo);
	//����ͼƬ
	cv::imwrite(filename, frame);

	//imshow(kWindows, frame);
	//imshow(imgname, frame);
	cv::waitKey(0);
	
	//������������⵽������: �����˱��-λ�ñ��-���-�ļ���
	//sendInfo(filename, sortedName);

	//���
	sortedName.clear();
}


//�������Ŷ���ֵȥ�������Ŷ�Ԥ���
void postprocess(Mat& frame,const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i) 
	{
		//�������������еĿ򣬲��ҳ����Ŷ���ߵĿ�
		//�������Ŷ���ߵ����ֵ�������
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold) 
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int high = (int)(data[3] * frame.rows);
				int left = centerX - (width / 2);
				int top = centerY - (high / 2);

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, high));
			}
		}
	}

	//�Ǽ���ֵ����
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	vector<vector<int> > info;

	for (size_t i = 0; i < indices.size(); ++i)
	{	
		vector<int> vet;
		int idx = indices[i];
		Rect box = boxes[idx];
		//��ǳ����岢��ʾ�������Ŷ�
		drawPred(classIds[idx], confidences[idx], 
			box.x, box.y, box.x + box.width, box.y + box.height, frame);
		//��ͼƬ�м�⵽���Ŷ���ߵ��������id���Ͻ����걣��
		string label;
		label = classes[classIds[idx]];
		//cout << label << endl;
		vet.push_back(box.x);
		vet.push_back(box.y);
		vet.push_back(box.width);
		vet.push_back(box.height);
		vet.push_back(classIds[idx]);
		info.push_back(vet);
	}
	
	getInfo(info);	
}


//����Ԥ���
void drawPred(int classId, float confidence, int left, int top, int right, int bottom, Mat& frame)
{

	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(50, 50, 255), 1);

	string label;
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId];

		//cout << label << endl;

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		rectangle(frame, Point(left, top - round(1.5*labelSize.height)),
			Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
		putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 2);
	}
}


//��������
vector<String> getOutputsNames(const Net& net) {
	static vector<String> names;
	if (names.empty()) 
	{
		//�������������
		vector<int> outLayers = net.getUnconnectedOutLayers();
		//����������
		vector<String> layersname = net.getLayerNames();
		//��������������������
		names.resize(layersname.size());
		for (size_t i = 0; i < outLayers.size(); ++i )
		{
			names[i] = layersname[outLayers[i] - 1];
		}
	}
	return names;
}


/*����������������Ͻ�X�����С��˳��ƴ�ӳ��ַ���
	������м���_���ż�����������ֿ���һ��Ŀ�꣩*/
void getInfo(vector<vector<int> > info)
{	
	string nameSort;
	
	//����
	std::sort(info.begin(), info.end());
	for (int i = 0; i < (info.size()); i++)
	{	
		//���������Ŀ������ֱ��ƴ��(��¯��)��������ӡ�_�����ŷָ�
		string classname;
		int X_redaction = 0;
		int Y_redaction = 0;
		
		if (info.size() == 1) {
			classname = format("%s", classes[info[i][4]]);
		}
		else{ 
			if (i+1 == info.size()) {
				classname = format("%s", classes[info[i][4]]);
			}			
			else
			{	
				X_redaction = abs(info[i + 1][0] - info[i][0]);
				Y_redaction = abs(info[i][1] - info[i + 1][1]);
				if (X_redaction < info[i][2] +10 && Y_redaction < 5) 
				{
					classname = format("%s", classes[info[i][4]]);
				}		
				else
				{
					classname = format("%s_", classes[info[i][4]]);
				}
			}
		}
		sortedName.append(classname);		
	}
	cout << sortedName << endl;
}


//�������ݿ���д洢���߲�ѯ
void loadDatabase() {
	//��������
	MYSQL mysql, *conn;
	const char *host = "";
	const char *user = "pigff";
	const char *pwd = "pigff520";
	const char *db = "test";
	unsigned int port = 3306;
	const char *unix_scoket = NULL;
	unsigned long client_flag = 0;


	MYSQL_RES *result;		//���
	MYSQL_ROW row;			//�����_��

	//��ʼ�����Ӿ��
	conn = mysql_init(NULL);
	//mysql_init(&mysql);
	//�������ݿ�
	if ((conn = mysql_real_connect(&mysql, host, user, pwd, db, port, unix_scoket, client_flag)) == NULL) {
		printf("fail to connect mysql \n");
		fprintf(stderr, " %s\n", mysql_error(&mysql));
		exit(1);
	}
	else
	{
		printf("connect ok!! \n");
	}
	//ִ��sql�ű�
	if (mysql_query(&mysql, "")!=0)
	{
		fprintf(stderr, "fail to query! \n");
		exit(1);
	}
	else
	{
		if ((result = mysql_store_result(&mysql)) == NULL) {
			fprintf(stderr, "fail to store result!\n");
			exit(1);
		}
	}
	
	mysql_free_result(result); //�ͷŽ���� 
	mysql_close(conn); //�ر����� 
	system("pause");
	exit(EXIT_SUCCESS);

}


//������Ƶ����ͼƬ��·��
string get_time(bool savevideo) {
	//�����ļ���ʱ������
	time_t nowtime = time(NULL);
	struct tm *p;
	//p = gmtime(&nowtime);
	p = localtime(&nowtime);
	char filename[256] = { 0 };
	char imgname[256] = { 0 };
	sprintf(imgname, "%.2d_%.2d_%.2d_%.2d_%.2d", 1 + p->tm_mon,
		p->tm_mday, ((p->tm_hour)), p->tm_min, p->tm_sec);
	//����Ƶ�����ڹ���/data/vedio�ļ����£����ͼƬ������/data/result�ļ�����
	if (savevideo) {
		sprintf(filename, ".\\data\\vedio/%d_%.2d_%.2d_%.2d_%.2d_%.2d.mp4", 1900 +
			p->tm_year, 1 + p->tm_mon, p->tm_mday, ((p->tm_hour)), p->tm_min, p->tm_sec);
	}
	else
	{
		sprintf(filename, ".\\data\\results/%d_%.2d_%.2d_%.2d_%.2d_%.2d.jpg", 1900 +
			p->tm_year, 1 + p->tm_mon, p->tm_mday, ((p->tm_hour)), p->tm_min, p->tm_sec);
	}
	

	cout << filename << endl;
	return filename;
}


/*�������׼��Ϣ�����������ܵ���������λ��
	���������ͼƬ·������������Խ��*/
void sendInfo(string imgDir, string detectResults) {

	vector<string> position;						//��¯��ţ�λ�ã�
	vector<string> result1;							//��������״̬
	string info;									//����õ���Ϣ
	char* delim = "_";								//�ָ���
	int count1 = 0;									//��⵽�ĸ���������
	int count2 = 0;									//��⵽�ķ�����
													//stringתΪchar*
	char* str = new char[detectResults.length() + 1];
	strcpy(str, detectResults.c_str());
	char * d = "_";
	//�ָ�
	char *p = strtok(str, d);
	while (p) {
		string s = p;						//�ָ���
		if (isdigit(s[0]) != 0)
		{
			position.push_back(s);
		}
		if (s == "close")
		{
			result1.push_back(s);
			count2++;
		}
		if (s == "open")
		{
			result1.push_back(s);
			count2++;
		}
		if (s == "columns")
		{
			result1.push_back(s);
			count1++;
		}
		if (s == "shackles")
		{
			result1.push_back(s);
			count1++;
		}
	}
	//δ��⵽��¯��ţ���������Ϣ����λ��
	if (position.size() == 0) return;

	//��⵽�����¯���
	if (position.size() > 1)
	{
		for (int i = 0; i < position.size(); ++i)
		{

		}
	}
	//ֻ��⵽һ����¯���
	if (position.size() == 1) {

	}
}


int main()
{
	//Yolo����
	string testImg1 = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\images\\000027.jpg";
	string testImg2 = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\1/images/gate000002.jpg";
	cv::Mat img;
	img = cv::imread(testImg1);
	detect_image(img,weight_JL,model_JL,class_JL);
	//Open_Camera("12", "34", "56", "78");
}