// ***************************************
// tingjian liu
// 2016/1/29
// vs2015
// 采用最小错误率的贝叶斯决策来处理irirs数据集的分类问题
// ***************************************

#define	_CRT_SECURE_NO_WARNINGS
#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#include	<math.h>

// ***************************************
// 常量
// ***************************************
#define	CLASS_CNT	3	// 分类的类型种数
#define FEATURE_VECTOR_SIZE	4	// 每个样本的特征数
#define	SAMPLE_CNT	150	// 样本总数

// ***************************************
// 样本类型定义
// ***************************************
typedef	struct iris_sample_type
{
	char	label[20];	// 样本的类别标签，包括Iris-setosa,Iris-versicolor,Iris-virginica
	double		fv[FEATURE_VECTOR_SIZE];	// 每个样本包含的四个特征的值
}IRIRSSample;

// ***************************************
// 全局变量声明
// ***************************************
static	IRIRSSample iris[SAMPLE_CNT]; // 样本特征库
static	double		mean[CLASS_CNT][FEATURE_VECTOR_SIZE];	// 每个类的各个特征向量的均值
static	double		cov[CLASS_CNT][FEATURE_VECTOR_SIZE][FEATURE_VECTOR_SIZE]; // 各个类的FEATURE_VECTOR_SIZE*FEATURE_VECTOR_SIZE维协方差矩阵
static	double		covI[CLASS_CNT][FEATURE_VECTOR_SIZE][FEATURE_VECTOR_SIZE]; // cov的逆矩阵
static	double		det[CLASS_CNT];		// 矩阵cov的行列式
const	int		idx1[CLASS_CNT] = { 0, 50, 100 };
const	int		idx2[CLASS_CNT] = { 49, 99, 149 };

// ***************************************
// 函数原型
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
// 主函数
// ***************************************
int main() {
	// 加载数据
	loadData("iris.data");

	// 留一交叉验证法
	leaveOneOut();

	return 0;
}

// ***************************************
// 加载数据
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
// 留一交叉验证法：只使用原本样本中的一项来当做验证资料， 
// 而剩余的则留下来当做训练资料。 这个步骤一直持续到每个样本都被当做一次验证资料
// ***************************************
void leaveOneOut() {
	char	label[20];
	int		ci;		// 第i类
	int		correctCnt = 0;
	int		i;

	for ( ci = 0; ci < CLASS_CNT; ci++)
	{
		// 计算所有类别的统计量
		getStatistics();

		// 交叉验证
		for ( i = idx1[ci]; i <= idx2[ci]; i++)
		{
			// 重新计算扣除第ci类的第i个样本后的所有统计量
			getLeaveOneStatistics(ci, i);
			printf("%3d ", i);
			// 用贝叶斯分类器对第i个样本进行分类
			bayesClassifier(i, label);
			
			// 验证分类的结果
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

	printf("\n成功率：%lf%% \n", 100*(correctCnt / (SAMPLE_CNT*1.00)));
}

// ***************************************
// 计算所需的统计量
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
// 计算各个类的均值向量
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
			fv[j] = 0;	// 重置
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
// 计算各个类的协方差矩阵
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
// 计算协方差矩阵cov的逆矩阵
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
// 计算协方差矩阵cov的行列式
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
// 计算扣除第ci类第i个样本后的ci类的所有统计量
// ***************************************
void getLeaveOneStatistics(int ci, int leave) {
	getLeaveOneMeanVector(ci, leave);
	getLeaveOneCovMatrix(ci, leave);
	inverse(cov[ci], covI[ci]);
	determinant(cov[ci], &det[ci]);
}

// ***************************************
// 计算扣除第ci类第i个样本后的ci类的均值向量
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
		//printf("扣除一个后第%d类的均值为: %lf \n", ci, mean[ci][j]);
	}
}

// ***************************************
// 计算扣除第ci类第i个样本后的ci类的协方差矩阵
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
			// 无偏估计求协方差的分母为样本数减一	
			cov[ci][u][v] = s / (end - start - 1);
		}
	}
}

void bayesClassifier(int leave, char* label) {
	int		i, j, ci;
	double	fv[FEATURE_VECTOR_SIZE];
	double	fvSubM[FEATURE_VECTOR_SIZE];
	double	t[FEATURE_VECTOR_SIZE];
	double	dm[CLASS_CNT];		// 马氏距离
	double	gi[CLASS_CNT];		// 每个类的决策函数值
	double	maxGi = -100000000.0;				// 最大的决策函数值
	int		maxCi = 0;			// maxGi的对应的类
	
	for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
	{
		fv[i] = iris[leave].fv[i];
	}

	// 计算第leave样本的各个类的判别函数值
	for ( ci = 0; ci < CLASS_CNT; ci++)
	{
		// 计算向量： fv - mean
		for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
		{
			fvSubM[i] = fv[i] - mean[ci][i];
		}
		// 计算向量：(fv - mean)*covI
		for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
		{
			t[i] = 0.0;
			for ( j = 0; j < FEATURE_VECTOR_SIZE; j++)
			{
				t[i] += fvSubM[j] * covI[ci][j][i];
			}
		}
		// 计算向量：(fv - mean)^t*covI*(fv - mean)
		// 即是计算待识别样本点到每类的均值点的马氏距离
		dm[ci] = 0.0;
		for ( i = 0; i < FEATURE_VECTOR_SIZE; i++)
		{
			dm[ci] += t[i] * fvSubM[i];
		}
		// 决策函数,因为每个类的样本数所占比例相同，所以忽略
		gi[ci] = -0.5*dm[ci] - log(sqrt(det[ci]));
		printf("  g[%d] = %-12.6lf", ci, gi[ci]);
	}
	
	// 找出最大的判决函数值所对应的类
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