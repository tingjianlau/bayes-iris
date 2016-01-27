#define	_CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>

using namespace std;

// *****************************************************
// Constants and Type Definitions
// *****************************************************

#define	ClassNum			3
#define	FeatureVectorSize	4
#define	SampleNum			150

typedef struct iris_sample_type {
	char	label[16];
	int		id;
	int		fv[FeatureVectorSize];
} IRISSample;

// *****************************************************
// Function Prototypes
// *****************************************************

static void load_iris_data();
static void leave_one_out_test();
static void estimate();
static void estimate_mean_vector();
static void estimate_covariance_matrix();
static void estimate_mean_vector_for_iris_leave_one_out(int ci, int idx1, int idx2, int leave);
static void estimate_covariance_matrix_for_iris_leave_one_out(int ci, int idx1, int idx2, int leave);
static void bayes_classifier(int idx, char label[16]);
static void output_iris_parameter();
static void inverse(double s[4][4], double I[4][4]);
static void determinant(double s[4][4], double& det);

// *****************************************************
// Global Data
// *****************************************************

// �ӱ������
static IRISSample iris[SampleNum];

// �yӋ����
static double m[ClassNum][FeatureVectorSize];
static double C[ClassNum][FeatureVectorSize][FeatureVectorSize];

// CI: C �ķ����
static double CI[ClassNum][FeatureVectorSize][FeatureVectorSize];
// dd: C ������ʽֵ
static double dd[ClassNum];

// *****************************************************
// Main routine
// *****************************************************

int main(int argc, char *argv[])
{

	// ��ȡ���е� IRIS Data
	load_iris_data();

	// Leave-one-out Test
	leave_one_out_test();

	system("PAUSE");
	return 0;

}

// *****************************************************
// load_iris_data
// Load all sample of IRIS data set
// *****************************************************

void load_iris_data()
{
	FILE *fp;
	char label[16];
	int id;
	int v0, v1, v2, v3;
	int n;

	if ((fp = fopen("iris.txt", "r")) == NULL) {
		cout << "Can not open file iris.txt!" << endl;
		exit(1);
	}

	n = 0;
	while (!feof(fp)) {
		fscanf(fp, "%d%d%d%d%s%d\n", &v0, &v1, &v2, &v3, label, &id);
		strcpy(iris[n].label, label);
		iris[n].id = id;
		iris[n].fv[0] = v0;
		iris[n].fv[1] = v1;
		iris[n].fv[2] = v2;
		iris[n].fv[3] = v3;
		n++;
	}

	cout << "Total sample # : " << n << endl;

	fclose(fp);

}

// *****************************************************
// leave_one_out_test
// Leave-one-out test
// *****************************************************

void leave_one_out_test()
{
	char label[16];
	int ci;
	const int idx1[ClassNum] = { 0, 50, 100 };
	const int idx2[ClassNum] = { 49, 99, 149 };
	int correct;
	int i;

	// ���_���R�Ęӱ����������
	correct = 0;

	// ci=0 (Setosa), ci=1 (Versicolor), ci=2 (Virginica)
	for (ci = 0; ci <= 2; ci++) {
		// ��Ӌ����e�� mean �c covariance���KӋ�� covariance �� inverse �c determinant
		estimate();
		inverse(C[0], CI[0]);
		inverse(C[1], CI[1]);
		inverse(C[2], CI[2]);
		determinant(C[0], dd[0]);
		determinant(C[1], dd[1]);
		determinant(C[2], dd[2]);
		// ���R ci e��ÿһ���ӱ�
		for (i = idx1[ci]; i <= idx2[ci]; i++) {
			printf("%3d ", i);
			// �۳� ci e�ĵ� i ���ӱ������¹�Ӌ mean �c covariance��Ӌ�� inverse �c determinant
			estimate_mean_vector_for_iris_leave_one_out(ci, idx1[ci], idx2[ci], i);
			estimate_covariance_matrix_for_iris_leave_one_out(ci, idx1[ci], idx2[ci], i);
			inverse(C[ci], CI[ci]);
			determinant(C[ci], dd[ci]);
			// ���� Bayes Classifier ���R�� i ���ӱ������R�Y�������� Label
			bayes_classifier(i, label);
			// ��C���R�Y����ָ�� Label �c���y�ӱ� i �Ƿ���ͬ
			if (strcmp(label, iris[i].label) == 0) {
				// Label ��ͬ�t���_ӛ��ۼ�һ
				correct++;
			}
			else {
				cout << "*";
			}
			cout << endl;
		}
	}

	// �@ʾ���R��
	printf("Recognition rate (leave-one-out test) = %6.2f %%\n", 100.0*correct / 150);

}

// *****************************************************
// estimate
// Estimate statistical parameters of IRIS samples
// *****************************************************

void estimate()
{

	estimate_mean_vector();
	estimate_covariance_matrix();

}

// *****************************************************
// estimate_mean_vector
// Estimate mean vectors of IRIS samples
// *****************************************************

void estimate_mean_vector()
{
	double s0, s1, s2, s3;
	int i;

	s0 = s1 = s2 = s3 = 0.0;
	for (i = 0; i <= 49; i++) {
		s0 += iris[i].fv[0];
		s1 += iris[i].fv[1];
		s2 += iris[i].fv[2];
		s3 += iris[i].fv[3];
	}
	m[0][0] = s0 / 50.0;
	m[0][1] = s1 / 50.0;
	m[0][2] = s2 / 50.0;
	m[0][3] = s3 / 50.0;

	s0 = s1 = s2 = s3 = 0.0;
	for (i = 50; i <= 99; i++) {
		s0 += iris[i].fv[0];
		s1 += iris[i].fv[1];
		s2 += iris[i].fv[2];
		s3 += iris[i].fv[3];
	}
	m[1][0] = s0 / 50.0;
	m[1][1] = s1 / 50.0;
	m[1][2] = s2 / 50.0;
	m[1][3] = s3 / 50.0;

	s0 = s1 = s2 = s3 = 0.0;
	for (i = 100; i <= 149; i++) {
		s0 += iris[i].fv[0];
		s1 += iris[i].fv[1];
		s2 += iris[i].fv[2];
		s3 += iris[i].fv[3];
	}
	m[2][0] = s0 / 50.0;
	m[2][1] = s1 / 50.0;
	m[2][2] = s2 / 50.0;
	m[2][3] = s3 / 50.0;

}

// *****************************************************
// estimate_covariance_matrix
// Estimate covariance matrices of IRIS samples
// *****************************************************

void estimate_covariance_matrix()
{
	double s;
	double du, dv;
	int u, v;
	int i;

	for (u = 0; u <= 3; u++) {
		for (v = 0; v <= 3; v++) {
			s = 0.0;
			for (i = 0; i <= 49; i++) {
				du = (iris[i].fv[u] - m[0][u]);
				dv = (iris[i].fv[v] - m[0][v]);
				s += (du * dv);
			}
			C[0][u][v] = s / 49.0;
		}
	}

	for (u = 0; u <= 3; u++) {
		for (v = 0; v <= 3; v++) {
			s = 0.0;
			for (i = 50; i <= 99; i++) {
				du = (iris[i].fv[u] - m[1][u]);
				dv = (iris[i].fv[v] - m[1][v]);
				s += (du * dv);
			}
			C[1][u][v] = s / 49.0;
		}
	}

	for (u = 0; u <= 3; u++) {
		for (v = 0; v <= 3; v++) {
			s = 0.0;
			for (i = 100; i <= 149; i++) {
				du = (iris[i].fv[u] - m[2][u]);
				dv = (iris[i].fv[v] - m[2][v]);
				s += (du * dv);
			}
			C[2][u][v] = s / 49.0;
		}
	}

}

// *****************************************************
// estimate_mean_vector_for_iris_leave_one_out
// Estimate mean vectors of IRIS samples
//   ci:	1, 2, and 3 for Setosa, Versicolor, and Viginica respectively
//   idx1:	starting index of samples
//   idx2:	ending index of samples
//   leave:	left index of samples
// *****************************************************

void estimate_mean_vector_for_iris_leave_one_out(int ci, int idx1, int idx2, int leave)
{
	double s0, s1, s2, s3;
	int i;

	// ����ۼ�ֵ
	s0 = s1 = s2 = s3 = 0.0;

	// �ۼ����� sample ������ֵ���۳� leave ��ָ���� sample��
	for (i = idx1; i <= idx2; i++) {
		if (i != leave) {
			// ���^ leave ��ָ���� sample
			s0 += iris[i].fv[0];
			s1 += iris[i].fv[1];
			s2 += iris[i].fv[2];
			s3 += iris[i].fv[3];
		}
	}

	// sample ������ idx2-idx1 (49)
	m[ci][0] = s0 / (idx2 - idx1);
	m[ci][1] = s1 / (idx2 - idx1);
	m[ci][2] = s2 / (idx2 - idx1);
	m[ci][3] = s3 / (idx2 - idx1);

}

// *****************************************************
// estimate_covariance_matrix_for_iris_leave_one_out
// Estimate covariance matrices of IRIS samples
//   ci:	1, 2, and 3 for Setosa, Versicolor, and Viginica respectively
//   idx1:	starting index of samples
//   idx2:	ending index of samples
//   leave:	left index of samples
// *****************************************************

void estimate_covariance_matrix_for_iris_leave_one_out(int ci, int idx1, int idx2, int leave)
{
	double s;
	double du, dv;
	int u, v;
	int i;

	for (u = 0; u <= 3; u++) {
		for (v = 0; v <= 3; v++) {
			// ����ۼ�ֵ
			s = 0.0;
			// �ۼ����� sample �����繲׃��ֵ���۳� leave ��ָ���� sample��
			for (i = idx1; i <= idx2; i++) {
				// ���^ leave ��ָ���� sample
				if (i != leave) {
					du = (iris[i].fv[u] - m[ci][u]);
					dv = (iris[i].fv[v] - m[ci][v]);
					s += (du * dv);
				}
			}
			// sample ������ idx2-idx1 (49)��unbiased estimate ������ idx2-idx1-1
			C[ci][u][v] = s / (idx2 - idx1 - 1);
		}
	}

}

// *****************************************************
// bayes_classifier
// Classify the i-th sample of IRIS samples by Bayes Classifier
// *****************************************************

void bayes_classifier(int idx, char label[16])
{
	double fv[FeatureVectorSize];
	double x[FeatureVectorSize];
	double t[FeatureVectorSize];
	double dm[ClassNum];
	double g[ClassNum];
	double max_g;
	int max_ci;
	int ci;
	int i, j;

	// ȡ�Øӱ� idx ����������
	for (i = 0; i <= FeatureVectorSize - 1; i++) {
		fv[i] = iris[idx].fv[i];
	}

	// Ӌ��ӱ� idx �cÿһ��e ci ���a�e����ֵ
	for (ci = 0; ci <= 2; ci++) {
		// x : v - m
		for (i = 0; i <= FeatureVectorSize - 1; i++) {
			x[i] = fv[i] - m[ci][i];
		}
		// t : (v - m)^t * CI
		for (i = 0; i <= FeatureVectorSize - 1; i++) {
			t[i] = 0.0;
			for (j = 0; j <= FeatureVectorSize - 1; j++) {
				t[i] += x[j] * CI[ci][j][i];
			}
		}
		// dm : (v - m)^t * CI * (v - m)
		// dm �������^�� Mahalanobis Distance
		dm[ci] = 0.0;
		for (i = 0; i <= FeatureVectorSize - 1; i++) {
			dm[ci] += t[i] * x[i];
		}
		// g : discriminant function
		g[ci] = -0.5 * dm[ci] - log(sqrt(dd[ci]));
		printf(" g[%d] = %12.6f", ci, g[ci]);
	}

	// �ҳ��a�e����ֵ�����
	max_g = -999999999.9;
	max_ci = 0;
	for (ci = 0; ci <= 2; ci++) {
		if (g[ci] > max_g) {
			max_g = g[ci];
			max_ci = ci;
		}
	}

	// �������R�Y�� 
	strcpy(label, iris[max_ci * 50].label);

}

// *****************************************************
// output_iris_parameter
// Output statistical parameters of IRIS samples
// *****************************************************

void output_iris_parameter()
{
	int c;
	int i, j;

	for (c = 0; c <= ClassNum - 1; c++) {
		printf("%s\n", iris[c * 50].label);
		printf("m");
		for (i = 0; i <= FeatureVectorSize - 1; i++) {
			printf(" %16.8f", m[c][i]);
		}
		printf("\n");
		for (i = 0; i <= FeatureVectorSize - 1; i++) {
			printf("C");
			for (j = 0; j <= FeatureVectorSize - 1; j++) {
				printf(" %16.8f", C[c][i][j]);
			}
			printf("\n");
		}
	}

}

// *****************************************************
// inverse
// Inverse of a invertible matrix
// *****************************************************

void inverse(double s[4][4], double I[4][4])
{

	// ***** Maple Code
	//
	// with(linalg);
	// Sigma := array( [[s00, s01, s02, s03],[s10,s11,s12,s13],[s20,s21,s22,s23],[s30,s31,s32,s33]] );
	// II := inverse(Sigma);
	// with(codegen);
	// C(II);
	//
	// ***** Replace II as I and sij as s[i][j]

	I[0][0] = (-s[1][1] * s[2][2] * s[3][3] + s[1][1] * s[2][3] * s[3][2] + s[2][1] * s[1][2] * s[3][3] - s[2][1] * s[1][3] * s[3][2] - s[3][1] * s[1][2] * s[2][3] +
		s[3][1] * s[1][3] * s[2][2]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[0][1] = -(-s[0][1] * s[2][2] * s[3][3] + s[0][1] * s[2][3] * s[3][2] + s[2][1] * s[0][2] * s[3][3] - s[2][1] * s[0][3] * s[3][2] - s[3][1] * s[0][2] * s[2][3]
		+ s[3][1] * s[0][3] * s[2][2]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2]
			- s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] *
			s[3][3] + s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3]
			* s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] *
			s[1][2] * s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1]
			* s[0][3] * s[1][2]);
	I[0][2] = (-s[0][1] * s[1][2] * s[3][3] + s[0][1] * s[1][3] * s[3][2] + s[1][1] * s[0][2] * s[3][3] - s[1][1] * s[0][3] * s[3][2] - s[3][1] * s[0][2] * s[1][3] +
		s[3][1] * s[0][3] * s[1][2]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[0][3] = -(-s[0][1] * s[1][2] * s[2][3] + s[0][1] * s[1][3] * s[2][2] + s[1][1] * s[0][2] * s[2][3] - s[1][1] * s[0][3] * s[2][2] - s[2][1] * s[0][2] * s[1][3]
		+ s[2][1] * s[0][3] * s[1][2]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2]
			- s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] *
			s[3][3] + s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3]
			* s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] *
			s[1][2] * s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1]
			* s[0][3] * s[1][2]);
	I[1][0] = -(-s[1][0] * s[2][2] * s[3][3] + s[1][0] * s[2][3] * s[3][2] + s[2][0] * s[1][2] * s[3][3] - s[2][0] * s[1][3] * s[3][2] - s[3][0] * s[1][2] * s[2][3]
		+ s[3][0] * s[1][3] * s[2][2]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2]
			- s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] *
			s[3][3] + s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3]
			* s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] *
			s[1][2] * s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1]
			* s[0][3] * s[1][2]);
	I[1][1] = (-s[0][0] * s[2][2] * s[3][3] + s[0][0] * s[2][3] * s[3][2] + s[2][0] * s[0][2] * s[3][3] - s[2][0] * s[0][3] * s[3][2] - s[3][0] * s[0][2] * s[2][3] +
		s[3][0] * s[0][3] * s[2][2]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[1][2] = -(-s[0][0] * s[1][2] * s[3][3] + s[0][0] * s[1][3] * s[3][2] + s[1][0] * s[0][2] * s[3][3] - s[1][0] * s[0][3] * s[3][2] - s[3][0] * s[0][2] * s[1][3]
		+ s[3][0] * s[0][3] * s[1][2]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2]
			- s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] *
			s[3][3] + s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3]
			* s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] *
			s[1][2] * s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1]
			* s[0][3] * s[1][2]);
	I[1][3] = -(s[0][0] * s[1][2] * s[2][3] - s[0][0] * s[1][3] * s[2][2] - s[1][0] * s[0][2] * s[2][3] + s[1][0] * s[0][3] * s[2][2] + s[2][0] * s[0][2] * s[1][3] -
		s[2][0] * s[0][3] * s[1][2]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[2][0] = (-s[1][0] * s[2][1] * s[3][3] + s[1][0] * s[2][3] * s[3][1] + s[2][0] * s[1][1] * s[3][3] - s[2][0] * s[1][3] * s[3][1] - s[3][0] * s[1][1] * s[2][3] +
		s[3][0] * s[1][3] * s[2][1]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[2][1] = -(-s[0][0] * s[2][1] * s[3][3] + s[0][0] * s[2][3] * s[3][1] + s[2][0] * s[0][1] * s[3][3] - s[2][0] * s[0][3] * s[3][1] - s[3][0] * s[0][1] * s[2][3]
		+ s[3][0] * s[0][3] * s[2][1]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2]
			- s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] *
			s[3][3] + s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3]
			* s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] *
			s[1][2] * s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1]
			* s[0][3] * s[1][2]);
	I[2][2] = (-s[0][0] * s[1][1] * s[3][3] + s[0][0] * s[1][3] * s[3][1] + s[1][0] * s[0][1] * s[3][3] - s[1][0] * s[0][3] * s[3][1] - s[3][0] * s[0][1] * s[1][3] +
		s[3][0] * s[0][3] * s[1][1]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[2][3] = (s[0][0] * s[1][1] * s[2][3] - s[0][0] * s[1][3] * s[2][1] - s[1][0] * s[0][1] * s[2][3] + s[1][0] * s[0][3] * s[2][1] + s[2][0] * s[0][1] * s[1][3] -
		s[2][0] * s[0][3] * s[1][1]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[3][0] = (s[1][0] * s[2][1] * s[3][2] - s[1][0] * s[2][2] * s[3][1] - s[2][0] * s[1][1] * s[3][2] + s[2][0] * s[1][2] * s[3][1] + s[3][0] * s[1][1] * s[2][2] -
		s[3][0] * s[1][2] * s[2][1]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[3][1] = -(s[0][0] * s[2][1] * s[3][2] - s[0][0] * s[2][2] * s[3][1] - s[2][0] * s[0][1] * s[3][2] + s[2][0] * s[0][2] * s[3][1] + s[3][0] * s[0][1] * s[2][2] -
		s[3][0] * s[0][2] * s[2][1]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[3][2] = (s[0][0] * s[1][1] * s[3][2] - s[0][0] * s[1][2] * s[3][1] - s[1][0] * s[0][1] * s[3][2] + s[1][0] * s[0][2] * s[3][1] + s[3][0] * s[0][1] * s[1][2] -
		s[3][0] * s[0][2] * s[1][1]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);
	I[3][3] = -(s[0][0] * s[1][1] * s[2][2] - s[0][0] * s[1][2] * s[2][1] - s[1][0] * s[0][1] * s[2][2] + s[1][0] * s[0][2] * s[2][1] + s[2][0] * s[0][1] * s[1][2] -
		s[2][0] * s[0][2] * s[1][1]) / (-s[0][0] * s[1][1] * s[2][2] * s[3][3] + s[0][0] * s[1][1] * s[2][3] * s[3][2] + s[0][0] * s[2][1] * s[1][2] * s[3][3] - s[0][0] * s[2][1] * s[1][3] * s[3][2] -
			s[0][0] * s[3][1] * s[1][2] * s[2][3] + s[0][0] * s[3][1] * s[1][3] * s[2][2] + s[1][0] * s[0][1] * s[2][2] * s[3][3] - s[1][0] * s[0][1] * s[2][3] * s[3][2] - s[1][0] * s[2][1] * s[0][2] * s[3][3]
			+ s[1][0] * s[2][1] * s[0][3] * s[3][2] + s[1][0] * s[3][1] * s[0][2] * s[2][3] - s[1][0] * s[3][1] * s[0][3] * s[2][2] - s[2][0] * s[0][1] * s[1][2] * s[3][3] + s[2][0] * s[0][1] * s[1][3] *
			s[3][2] + s[2][0] * s[1][1] * s[0][2] * s[3][3] - s[2][0] * s[1][1] * s[0][3] * s[3][2] - s[2][0] * s[3][1] * s[0][2] * s[1][3] + s[2][0] * s[3][1] * s[0][3] * s[1][2] + s[3][0] * s[0][1] * s[1][2]
			* s[2][3] - s[3][0] * s[0][1] * s[1][3] * s[2][2] - s[3][0] * s[1][1] * s[0][2] * s[2][3] + s[3][0] * s[1][1] * s[0][3] * s[2][2] + s[3][0] * s[2][1] * s[0][2] * s[1][3] - s[3][0] * s[2][1] *
			s[0][3] * s[1][2]);

}

// *****************************************************
// determinant
// Inverse of a invertible matrix
// *****************************************************

void determinant(double s[4][4], double& det)
{

	// ***** Maple Code
	//
	// with(linalg);
	// Sigma := array( [[s00, s01, s02, s03],[s10,s11,s12,s13],[s20,s21,s22,s23],[s30,s31,s32,s33]] );
	// d := det(Sigma);
	// with(codegen);
	// C(d);
	//
	// ***** Replace t0 as det and sij as s[i][j]

	det = s[0][0] * s[1][1] * s[2][2] * s[3][3] - s[0][0] * s[1][1] * s[2][3] * s[3][2] - s[0][0] * s[2][1] * s[1][2] * s[3][3] + s[0][0] * s[2][1] * s[1][3] * s[3][2] + s[0][0] *
		s[3][1] * s[1][2] * s[2][3] - s[0][0] * s[3][1] * s[1][3] * s[2][2] - s[1][0] * s[0][1] * s[2][2] * s[3][3] + s[1][0] * s[0][1] * s[2][3] * s[3][2] + s[1][0] * s[2][1] * s[0][2] * s[3][3] - s[1][0]
		* s[2][1] * s[0][3] * s[3][2] - s[1][0] * s[3][1] * s[0][2] * s[2][3] + s[1][0] * s[3][1] * s[0][3] * s[2][2] + s[2][0] * s[0][1] * s[1][2] * s[3][3] - s[2][0] * s[0][1] * s[1][3] * s[3][2] -
		s[2][0] * s[1][1] * s[0][2] * s[3][3] + s[2][0] * s[1][1] * s[0][3] * s[3][2] + s[2][0] * s[3][1] * s[0][2] * s[1][3] - s[2][0] * s[3][1] * s[0][3] * s[1][2] - s[3][0] * s[0][1] * s[1][2] * s[2][3]
		+ s[3][0] * s[0][1] * s[1][3] * s[2][2] + s[3][0] * s[1][1] * s[0][2] * s[2][3] - s[3][0] * s[1][1] * s[0][3] * s[2][2] - s[3][0] * s[2][1] * s[0][2] * s[1][3] + s[3][0] * s[2][1] * s[0][3] *
		s[1][2];

}