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

// 参数初始化
float confThreshold = 0.7;						// 置信度阈值
float nmsThreshold = 0.7;						// 极大值抑制阈值
int inpWidth = 416;								// 输入图片宽度
int inpHeight = 416;							// 输入图片高度
int savevideo = 0;								// 保存视频或结果图片（0为保存图片结果，1为保存视频结果）
bool isdetect = true;							// 是否进行检测
string cameraId = "1";							// 摄像机编号

//全局变量
long currentFrame = 1;							//当前帧数
char imgPath[20];
vector<string> classes;
string sortedName = "";							//排序好的检测信息：机器人编号-位置编号-检测结果-图片文件名
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



//-------------------------------------------摄像机-------------------------------------------
void Open_Camera(string user, string pwd, string ip, string port, bool isdetect) 
{
	/*
	参数：
	string user:相机用户名
	string pwd:	用户密码
	string ip:	相机ip地址
	string port 端口号
	*/
	//rstp流获得网络相机
	String dir1 = format
	("rtsp://%s:%s@%s:%s/MPEG-4/ch1/main/av_stream", user, pwd, ip, port);
	cv::VideoCapture cap("E:\\data\\test\\car.mp4");
	VideoWriter outputvideo;
	Mat frame;
	Mat videoframe;
	//是否保存视频文件
	if (savevideo) {
		//打开网络摄像头
		cv::VideoCapture Cap0(dir1);
		//保存视频路劲（时间命名）
		string filename = get_time(savevideo);
		//设置保存视频格式宽高
		cv::Size s = cv::Size((int)cap.get
			(CV_CAP_PROP_FRAME_WIDTH),(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

		outputvideo.open(filename, -1, 25.0, s, true);
		//逐帧保存视频
		while (true) {
			Cap0 >> videoframe;
			if (videoframe.empty()) break;
			cv::imshow("output", videoframe);
			//写入视频
			outputvideo << videoframe;
			if (char(waitKey(1)) == 'q') break;
		}
	}
	while (isdetect) {
		
		cap >> frame;
		//摄像头是否打开
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
	参数包括：
	Mat frame: 图片的Mat矩阵
	string modelWeights: 模型权重文件路径
	string modelConfiguration: 模型配置文件路径
	string classesFile: 类别文件路劲
	*/
	//逐行加载类别文件中的类名
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	//加载网络
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);					

	//输入图片/视频或者摄像头；输出文件路径
	string str, outputFile;
	VideoWriter video;
	//创建窗口
	static const string kWindows = "Deep Learning object detection in OpenCV";
	namedWindow(kWindows, WINDOW_NORMAL);

	Mat blob;
	blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight),Scalar(0, 0, 0), true, false);

	//网络进行输入
	net.setInput(blob);

	//向前传递网络并在输出层获得输出
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	//去掉置信度的框
	postprocess(frame, outs);

	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame :  %.2f ms", t);
	putText(frame, label, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));

	//保存文件以时间命名
	string filename = get_time(savevideo);
	//保存图片
	cv::imwrite(filename, frame);

	//imshow(kWindows, frame);
	//imshow(imgname, frame);
	cv::waitKey(0);
	
	//向服务器传输检测到的内容: 机器人编号-位置编号-结果-文件名
	//sendInfo(filename, sortedName);

	//清除
	sortedName.clear();
}


//利用置信度阈值去掉低置信度预测框
void postprocess(Mat& frame,const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i) 
	{
		//搜索网络中所有的框，并找出置信度最高的框。
		//将其置信度最高的类别赋值给这个框
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

	//非极大值抑制
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	vector<vector<int> > info;

	for (size_t i = 0; i < indices.size(); ++i)
	{	
		vector<int> vet;
		int idx = indices[i];
		Rect box = boxes[idx];
		//标记出物体并显示类别和置信度
		drawPred(classIds[idx], confidences[idx], 
			box.x, box.y, box.x + box.width, box.y + box.height, frame);
		//将图片中检测到置信度最高的物体类别id左上角坐标保存
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


//画出预测框
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


//获得输出层
vector<String> getOutputsNames(const Net& net) {
	static vector<String> names;
	if (names.empty()) 
	{
		//获得输出层的索引
		vector<int> outLayers = net.getUnconnectedOutLayers();
		//获得输出层名
		vector<String> layersname = net.getLayerNames();
		//利用索引获得输出层名称
		names.resize(layersname.size());
		for (size_t i = 0; i < outLayers.size(); ++i )
		{
			names[i] = layersname[outLayers[i] - 1];
		}
	}
	return names;
}


/*结果按检测物体的左上角X坐标大小按顺序拼接成字符串
	（类别中间用_符号间隔，相邻数字看作一个目标）*/
void getInfo(vector<vector<int> > info)
{	
	string nameSort;
	
	//排序
	std::sort(info.begin(), info.end());
	for (int i = 0; i < (info.size()); i++)
	{	
		//将距离近的目标名字直接拼接(焦炉号)其他类别间加“_”符号分割
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


//加载数据库进行存储或者查询
void loadDatabase() {
	//基本参数
	MYSQL mysql, *conn;
	const char *host = "";
	const char *user = "pigff";
	const char *pwd = "pigff520";
	const char *db = "test";
	unsigned int port = 3306;
	const char *unix_scoket = NULL;
	unsigned long client_flag = 0;


	MYSQL_RES *result;		//结果
	MYSQL_ROW row;			//结果集_行

	//初始化连接句柄
	conn = mysql_init(NULL);
	//mysql_init(&mysql);
	//连接数据库
	if ((conn = mysql_real_connect(&mysql, host, user, pwd, db, port, unix_scoket, client_flag)) == NULL) {
		printf("fail to connect mysql \n");
		fprintf(stderr, " %s\n", mysql_error(&mysql));
		exit(1);
	}
	else
	{
		printf("connect ok!! \n");
	}
	//执行sql脚本
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
	
	mysql_free_result(result); //释放结果集 
	mysql_close(conn); //关闭连接 
	system("pause");
	exit(EXIT_SUCCESS);

}


//保存视频或结果图片的路径
string get_time(bool savevideo) {
	//保存文件以时间命名
	time_t nowtime = time(NULL);
	struct tm *p;
	//p = gmtime(&nowtime);
	p = localtime(&nowtime);
	char filename[256] = { 0 };
	char imgname[256] = { 0 };
	sprintf(imgname, "%.2d_%.2d_%.2d_%.2d_%.2d", 1 + p->tm_mon,
		p->tm_mday, ((p->tm_hour)), p->tm_min, p->tm_sec);
	//将视频保存在工程/data/vedio文件夹下，结果图片保存在/data/result文件夹下
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


/*整理出标准信息并利用匿名管道发送至上位机
	参数：结果图片路径，检测类别粗略结果*/
void sendInfo(string imgDir, string detectResults) {

	vector<string> position;						//焦炉编号（位置）
	vector<string> result1;							//钢柱风门状态
	string info;									//排序好的信息
	char* delim = "_";								//分隔符
	int count1 = 0;									//检测到的钢柱铁链数
	int count2 = 0;									//检测到的风门数
													//string转为char*
	char* str = new char[detectResults.length() + 1];
	strcpy(str, detectResults.c_str());
	char * d = "_";
	//分割
	char *p = strtok(str, d);
	while (p) {
		string s = p;						//分割结果
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
	//未检测到焦炉编号，不反馈信息至上位机
	if (position.size() == 0) return;

	//检测到多个焦炉编号
	if (position.size() > 1)
	{
		for (int i = 0; i < position.size(); ++i)
		{

		}
	}
	//只检测到一个焦炉编号
	if (position.size() == 1) {

	}
}


int main()
{
	//Yolo测试
	string testImg1 = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\images\\000027.jpg";
	string testImg2 = "E:\\code\\C\\Yolo_test2\\Yolo_test2\\data\\1/images/gate000002.jpg";
	cv::Mat img;
	img = cv::imread(testImg1);
	detect_image(img,weight_JL,model_JL,class_JL);
	//Open_Camera("12", "34", "56", "78");
}