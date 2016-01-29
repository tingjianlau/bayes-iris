// ***************************************
// tingjian liu
// 2016/1/29
// vs2015
// ������С�����ʵı�Ҷ˹����������irirs���ݼ��ķ�������
// ***************************************

#define	_CRT_SECURE_NO_WARNINGS
#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#include	<math.h>

// ***************************************
// ����
// ***************************************
#define	CLASS_CNT	3	// �������������
#define FEATURE_VECTOR_SIZE	4	// ÿ��������������
#define	SAMPLE_CNT	150	// ��������

// ***************************************
// �������Ͷ���
// ***************************************
typedef	struct iris_sample_type
{
	char	label[20];	// ����������ǩ������Iris-setosa,Iris-versicolor,Iris-virginica
	double		fv[FEATURE_VECTOR_SIZE];	// ÿ�������������ĸ�������ֵ
}IRIRSSample;

// ***************************************
// ȫ�ֱ�������
// ***************************************
static	IRIRSSample iris[SAMPLE_CNT]; // ����������
static	double		mean[CLASS_CNT][FEATURE_VECTOR_SIZE];	// ÿ����ĸ������������ľ�ֵ
static	double		cov[CLASS_CNT][FEATURE_VECTOR_SIZE][FEATURE_VECTOR_SIZE]; // �������FEATURE_VECTOR_SIZE*FEATURE_VECTOR_SIZEάЭ�������
static	double		covI[CLASS_CNT][FEATURE_VECTOR_SIZE][FEATURE_VECTOR_SIZE]; // cov�������
static	double		det[CLASS_CNT];		// ����cov������ʽ
const	int		idx1[CLASS_CNT] = { 0, 50, 100 };
const	int		idx2[CLASS_CNT] = { 49, 99, 149 };

// ***************************************
// ����ԭ��
// ***************************************
static void loadData(char* src);
static void	leaveOneOut();	
static void getStatistics();
static void	getMeanVector();
static void getCovMatrix();
static void inverse(double(*cov)[FEATURE_VECTOR_SIZE], double(*covI)[FEATURE_VECTOR_SIZE]);
static void determinant(double(*cov)[FEATURE_VECTOR_SIZE], double* det);
static void getLeaveOneStatistics(int ci, int leave);
static void getLeaveOneMeanVector(int ci, int leave);
static void getLeaveOneCovMatrix(int ci, int leave);
static void bayesClassifier(int leave, char* label);

// ***************************************
// ������
// ***************************************
int main() {
	// ��������
	loadData("iris.data");

	// ��һ������֤��
	leaveOneOut();

	return 0;
}

// ***************************************
// ��������
// ***************************************
void loadData(char* src) {
	FILE*	fp;
	char	label[20];
	double	fv[FEATURE_VECTOR_SIZE];
	int		n = 0, i;

	if ((fp = fopen(src, "r")) == NULL)
	{
		printf("Open file failed!\n");
		exit(0);
	}

	while (!feof(fp))
	{
		fscanf(fp, "%lf,%lf,%lf,%lf,%s\n", &fv[0], &fv[1], &fv[2], &fv[3], label);
		for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
		{
			iris[n].fv[i] = fv[i];
		}
		strcpy(iris[n].label, label);

		//printf("%lf,%lf,%lf,%lf,%s\n", iris[n].fv[0], iris[n].fv[1], iris[n].fv[2], iris[n].fv[3], iris[n].label);
		++n;
	}

	//printf("Total sample : %d \n", n);

	fclose(fp);
}

// ***************************************
// ��һ������֤����ֻʹ��ԭ�������е�һ����������֤���ϣ� 
// ��ʣ���������������ѵ�����ϡ� �������һֱ������ÿ��������������һ����֤����
// ***************************************
void leaveOneOut() {
	char	label[20];
	int		ci;		// ��i��
	int		correctCnt = 0;
	int		i;

	for ( ci = 0; ci < CLASS_CNT; ci++)
	{
		// ������������ͳ����
		getStatistics();

		// ������֤
		for ( i = idx1[ci]; i <= idx2[ci]; i++)
		{
			// ���¼���۳���ci��ĵ�i�������������ͳ����
			getLeaveOneStatistics(ci, i);
			printf("%3d ", i);
			// �ñ�Ҷ˹�������Ե�i���������з���
			bayesClassifier(i, label);
			
			// ��֤����Ľ��
			if (strcmp(label, iris[i].label) == 0)
			{
				printf("succ\n");
				correctCnt++;
			}
			else {
				printf("error\n");
			}
			
		}
	}

	printf("\n�ɹ��ʣ�%lf%% \n", 100*(correctCnt / (SAMPLE_CNT*1.00)));
}

// ***************************************
// ���������ͳ����
// ***************************************
void getStatistics() 
{
	int ci;

	//printf("\n");
	getMeanVector();
	getCovMatrix();

	for (ci = 0;  ci< CLASS_CNT; ci++)
	{
		inverse(cov[ci], covI[ci]);
		determinant(cov[ci], &det[ci]);
		//printf("\n%.12lf ", det[ci]);
	}
}

// ***************************************
// ���������ľ�ֵ����
// ***************************************
void getMeanVector() {
	double	fv[FEATURE_VECTOR_SIZE] = { 0 };
	int		i, j, ci;

	for ( ci = 0; ci < CLASS_CNT; ci++)
	{
		for ( i = idx1[ci]; i <= idx2[ci]; i++)
		{
			for ( j = 0; j < FEATURE_VECTOR_SIZE; j++)
			{
				fv[j] += iris[i].fv[j];
			}
		}
		for (j = 0; j < FEATURE_VECTOR_SIZE; j++)
		{
			mean[ci][j] = fv[j] / (SAMPLE_CNT / CLASS_CNT);
			fv[j] = 0;	// ����
		}
	}
	/*for (ci = 0; ci < CLASS_CNT; ci++) {
		for (j = 0; j < FEATURE_VECTOR_SIZE; j++)
		{
			printf("%lf ", mean[ci][j]);
		}
	}*/

}

// ***************************************
// ����������Э�������
// ***************************************
void getCovMatrix() {
	int		i, u, v, ci;
	double	du, dv, s;

	for ( ci = 0; ci < CLASS_CNT; ci++)
	{
		for ( u = 0; u < FEATURE_VECTOR_SIZE; u++)
		{
			for ( v = 0; v < FEATURE_VECTOR_SIZE; v++)
			{
				s = 0.0;
				for ( i = idx1[ci]; i <= idx2[ci]; i++)
				{
					du = iris[i].fv[u] - mean[ci][u];
					dv = iris[i].fv[v] - mean[ci][v];
					s += du*dv;
				}
				cov[ci][u][v] = s / (SAMPLE_CNT / CLASS_CNT - 1);
			}
		}
	}
	//printf("\n");
	/*for (ci = 0; ci < CLASS_CNT; ci++)
	{
		for (u = 0; u < FEATURE_VECTOR_SIZE; u++)
		{
			for (v = 0; v < FEATURE_VECTOR_SIZE; v++)
			{
				printf("%lf ", cov[ci][u][v]);
			}
		}
	}*/
}

// ***************************************
// ����Э�������cov�������
// ***************************************
void inverse(double(*cov)[FEATURE_VECTOR_SIZE], double(*covI)[FEATURE_VECTOR_SIZE]) {
	covI[0][0] = (-cov[1][1] * cov[2][2] * cov[3][3] + cov[1][1] * cov[2][3] * cov[3][2] + cov[2][1] * cov[1][2] * cov[3][3] - cov[2][1] * cov[1][3] * cov[3][2] - cov[3][1] * cov[1][2] * cov[2][3] +
		cov[3][1] * cov[1][3] * cov[2][2]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[0][1] = -(-cov[0][1] * cov[2][2] * cov[3][3] + cov[0][1] * cov[2][3] * cov[3][2] + cov[2][1] * cov[0][2] * cov[3][3] - cov[2][1] * cov[0][3] * cov[3][2] - cov[3][1] * cov[0][2] * cov[2][3]
		+ cov[3][1] * cov[0][3] * cov[2][2]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2]
			- cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] *
			cov[3][3] + cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3]
			* cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] *
			cov[1][2] * cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1]
			* cov[0][3] * cov[1][2]);
	covI[0][2] = (-cov[0][1] * cov[1][2] * cov[3][3] + cov[0][1] * cov[1][3] * cov[3][2] + cov[1][1] * cov[0][2] * cov[3][3] - cov[1][1] * cov[0][3] * cov[3][2] - cov[3][1] * cov[0][2] * cov[1][3] +
		cov[3][1] * cov[0][3] * cov[1][2]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[0][3] = -(-cov[0][1] * cov[1][2] * cov[2][3] + cov[0][1] * cov[1][3] * cov[2][2] + cov[1][1] * cov[0][2] * cov[2][3] - cov[1][1] * cov[0][3] * cov[2][2] - cov[2][1] * cov[0][2] * cov[1][3]
		+ cov[2][1] * cov[0][3] * cov[1][2]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2]
			- cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] *
			cov[3][3] + cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3]
			* cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] *
			cov[1][2] * cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1]
			* cov[0][3] * cov[1][2]);
	covI[1][0] = -(-cov[1][0] * cov[2][2] * cov[3][3] + cov[1][0] * cov[2][3] * cov[3][2] + cov[2][0] * cov[1][2] * cov[3][3] - cov[2][0] * cov[1][3] * cov[3][2] - cov[3][0] * cov[1][2] * cov[2][3]
		+ cov[3][0] * cov[1][3] * cov[2][2]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2]
			- cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] *
			cov[3][3] + cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3]
			* cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] *
			cov[1][2] * cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1]
			* cov[0][3] * cov[1][2]);
	covI[1][1] = (-cov[0][0] * cov[2][2] * cov[3][3] + cov[0][0] * cov[2][3] * cov[3][2] + cov[2][0] * cov[0][2] * cov[3][3] - cov[2][0] * cov[0][3] * cov[3][2] - cov[3][0] * cov[0][2] * cov[2][3] +
		cov[3][0] * cov[0][3] * cov[2][2]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[1][2] = -(-cov[0][0] * cov[1][2] * cov[3][3] + cov[0][0] * cov[1][3] * cov[3][2] + cov[1][0] * cov[0][2] * cov[3][3] - cov[1][0] * cov[0][3] * cov[3][2] - cov[3][0] * cov[0][2] * cov[1][3]
		+ cov[3][0] * cov[0][3] * cov[1][2]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2]
			- cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] *
			cov[3][3] + cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3]
			* cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] *
			cov[1][2] * cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1]
			* cov[0][3] * cov[1][2]);
	covI[1][3] = -(cov[0][0] * cov[1][2] * cov[2][3] - cov[0][0] * cov[1][3] * cov[2][2] - cov[1][0] * cov[0][2] * cov[2][3] + cov[1][0] * cov[0][3] * cov[2][2] + cov[2][0] * cov[0][2] * cov[1][3] -
		cov[2][0] * cov[0][3] * cov[1][2]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[2][0] = (-cov[1][0] * cov[2][1] * cov[3][3] + cov[1][0] * cov[2][3] * cov[3][1] + cov[2][0] * cov[1][1] * cov[3][3] - cov[2][0] * cov[1][3] * cov[3][1] - cov[3][0] * cov[1][1] * cov[2][3] +
		cov[3][0] * cov[1][3] * cov[2][1]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[2][1] = -(-cov[0][0] * cov[2][1] * cov[3][3] + cov[0][0] * cov[2][3] * cov[3][1] + cov[2][0] * cov[0][1] * cov[3][3] - cov[2][0] * cov[0][3] * cov[3][1] - cov[3][0] * cov[0][1] * cov[2][3]
		+ cov[3][0] * cov[0][3] * cov[2][1]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2]
			- cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] *
			cov[3][3] + cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3]
			* cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] *
			cov[1][2] * cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1]
			* cov[0][3] * cov[1][2]);
	covI[2][2] = (-cov[0][0] * cov[1][1] * cov[3][3] + cov[0][0] * cov[1][3] * cov[3][1] + cov[1][0] * cov[0][1] * cov[3][3] - cov[1][0] * cov[0][3] * cov[3][1] - cov[3][0] * cov[0][1] * cov[1][3] +
		cov[3][0] * cov[0][3] * cov[1][1]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[2][3] = (cov[0][0] * cov[1][1] * cov[2][3] - cov[0][0] * cov[1][3] * cov[2][1] - cov[1][0] * cov[0][1] * cov[2][3] + cov[1][0] * cov[0][3] * cov[2][1] + cov[2][0] * cov[0][1] * cov[1][3] -
		cov[2][0] * cov[0][3] * cov[1][1]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[3][0] = (cov[1][0] * cov[2][1] * cov[3][2] - cov[1][0] * cov[2][2] * cov[3][1] - cov[2][0] * cov[1][1] * cov[3][2] + cov[2][0] * cov[1][2] * cov[3][1] + cov[3][0] * cov[1][1] * cov[2][2] -
		cov[3][0] * cov[1][2] * cov[2][1]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[3][1] = -(cov[0][0] * cov[2][1] * cov[3][2] - cov[0][0] * cov[2][2] * cov[3][1] - cov[2][0] * cov[0][1] * cov[3][2] + cov[2][0] * cov[0][2] * cov[3][1] + cov[3][0] * cov[0][1] * cov[2][2] -
		cov[3][0] * cov[0][2] * cov[2][1]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[3][2] = (cov[0][0] * cov[1][1] * cov[3][2] - cov[0][0] * cov[1][2] * cov[3][1] - cov[1][0] * cov[0][1] * cov[3][2] + cov[1][0] * cov[0][2] * cov[3][1] + cov[3][0] * cov[0][1] * cov[1][2] -
		cov[3][0] * cov[0][2] * cov[1][1]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
	covI[3][3] = -(cov[0][0] * cov[1][1] * cov[2][2] - cov[0][0] * cov[1][2] * cov[2][1] - cov[1][0] * cov[0][1] * cov[2][2] + cov[1][0] * cov[0][2] * cov[2][1] + cov[2][0] * cov[0][1] * cov[1][2] -
		cov[2][0] * cov[0][2] * cov[1][1]) / (-cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] + cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] + cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] - cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] -
			cov[0][0] * cov[3][1] * cov[1][2] * cov[2][3] + cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] + cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] - cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] - cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3]
			+ cov[1][0] * cov[2][1] * cov[0][3] * cov[3][2] + cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] - cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] - cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] + cov[2][0] * cov[0][1] * cov[1][3] *
			cov[3][2] + cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] - cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] - cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] + cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] + cov[3][0] * cov[0][1] * cov[1][2]
			* cov[2][3] - cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] - cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] + cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] + cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] - cov[3][0] * cov[2][1] *
			cov[0][3] * cov[1][2]);
}

// ***************************************
// ����Э�������cov������ʽ
// ***************************************
void determinant(double(*cov)[FEATURE_VECTOR_SIZE], double* det) {
	*det = cov[0][0] * cov[1][1] * cov[2][2] * cov[3][3] - cov[0][0] * cov[1][1] * cov[2][3] * cov[3][2] - cov[0][0] * cov[2][1] * cov[1][2] * cov[3][3] + cov[0][0] * cov[2][1] * cov[1][3] * cov[3][2] + cov[0][0] *
		cov[3][1] * cov[1][2] * cov[2][3] - cov[0][0] * cov[3][1] * cov[1][3] * cov[2][2] - cov[1][0] * cov[0][1] * cov[2][2] * cov[3][3] + cov[1][0] * cov[0][1] * cov[2][3] * cov[3][2] + cov[1][0] * cov[2][1] * cov[0][2] * cov[3][3] - cov[1][0]
		* cov[2][1] * cov[0][3] * cov[3][2] - cov[1][0] * cov[3][1] * cov[0][2] * cov[2][3] + cov[1][0] * cov[3][1] * cov[0][3] * cov[2][2] + cov[2][0] * cov[0][1] * cov[1][2] * cov[3][3] - cov[2][0] * cov[0][1] * cov[1][3] * cov[3][2] -
		cov[2][0] * cov[1][1] * cov[0][2] * cov[3][3] + cov[2][0] * cov[1][1] * cov[0][3] * cov[3][2] + cov[2][0] * cov[3][1] * cov[0][2] * cov[1][3] - cov[2][0] * cov[3][1] * cov[0][3] * cov[1][2] - cov[3][0] * cov[0][1] * cov[1][2] * cov[2][3]
		+ cov[3][0] * cov[0][1] * cov[1][3] * cov[2][2] + cov[3][0] * cov[1][1] * cov[0][2] * cov[2][3] - cov[3][0] * cov[1][1] * cov[0][3] * cov[2][2] - cov[3][0] * cov[2][1] * cov[0][2] * cov[1][3] + cov[3][0] * cov[2][1] * cov[0][3] *
		cov[1][2];
}

// ***************************************
// ����۳���ci���i���������ci�������ͳ����
// ***************************************
void getLeaveOneStatistics(int ci, int leave) {
	getLeaveOneMeanVector(ci, leave);
	getLeaveOneCovMatrix(ci, leave);
	inverse(cov[ci], covI[ci]);
	determinant(cov[ci], &det[ci]);
}

// ***************************************
// ����۳���ci���i���������ci��ľ�ֵ����
// ***************************************
void getLeaveOneMeanVector(int ci, int leave) {
	double	fv[FEATURE_VECTOR_SIZE] = { 0 };
	int		i, j;
	int		start = idx1[ci], end = idx2[ci];

	for (i = start; i <= end; i++)
	{
		if (i != leave)
		{
			for (j = 0; j < FEATURE_VECTOR_SIZE; j++)
			{
				fv[j] += iris[i].fv[j];
			}
		}
	}
	for (j = 0; j < FEATURE_VECTOR_SIZE; j++)
	{
		mean[ci][j] = fv[j] / (end - start);
		//printf("�۳�һ�����%d��ľ�ֵΪ: %lf \n", ci, mean[ci][j]);
	}
}

// ***************************************
// ����۳���ci���i���������ci���Э�������
// ***************************************
void getLeaveOneCovMatrix(int ci, int leave) {
	int		i, u, v;
	double	du, dv, s;
	int		start = idx1[ci], end = idx2[ci];

	for (u = 0; u < FEATURE_VECTOR_SIZE; u++)
	{
		for (v = 0; v < FEATURE_VECTOR_SIZE; v++)
		{
			s = 0.0;
			for (i = start; i <= end; i++)
			{
				if (i != leave)
				{
					du = iris[i].fv[u] - mean[ci][u];
					dv = iris[i].fv[v] - mean[ci][v];
					s += du*dv;
				}
			}
			// ��ƫ������Э����ķ�ĸΪ��������һ	
			cov[ci][u][v] = s / (end - start - 1);
		}
	}
}

void bayesClassifier(int leave, char* label) {
	int		i, j, ci;
	double	fv[FEATURE_VECTOR_SIZE];
	double	fvSubM[FEATURE_VECTOR_SIZE];
	double	t[FEATURE_VECTOR_SIZE];
	double	dm[CLASS_CNT];		// ���Ͼ���
	double	gi[CLASS_CNT];		// ÿ����ľ��ߺ���ֵ
	double	maxGi = -100000000.0;				// ���ľ��ߺ���ֵ
	int		maxCi = 0;			// maxGi�Ķ�Ӧ����
	
	for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
	{
		fv[i] = iris[leave].fv[i];
	}

	// �����leave�����ĸ�������б���ֵ
	for ( ci = 0; ci < CLASS_CNT; ci++)
	{
		// ���������� fv - mean
		for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
		{
			fvSubM[i] = fv[i] - mean[ci][i];
		}
		// ����������(fv - mean)*covI
		for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
		{
			t[i] = 0.0;
			for ( j = 0; j < FEATURE_VECTOR_SIZE; j++)
			{
				t[i] += fvSubM[j] * covI[ci][j][i];
			}
		}
		// ����������(fv - mean)^t*covI*(fv - mean)
		// ���Ǽ����ʶ�������㵽ÿ��ľ�ֵ������Ͼ���
		dm[ci] = 0.0;
		for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
		{
			dm[ci] += t[i] * fvSubM[i];
		}
		// ���ߺ���,��Ϊÿ�������������ռ������ͬ�����Ժ���
		gi[ci] = -0.5*dm[ci] - log(sqrt(det[ci]));
		printf("  g[%d] = %-12.6lf", ci, gi[ci]);
	}
	
	// �ҳ������о�����ֵ����Ӧ����
	for ( ci = 0; ci < CLASS_CNT; ci++)
	{
		if (gi[ci] > maxGi)
		{
			maxGi = gi[ci];
			maxCi = ci;
		}
	}
	printf(" =%d= ", maxCi);
	strcpy(label, iris[maxCi * 50].label);
}