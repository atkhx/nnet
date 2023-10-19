package blas64

// enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
const (
	CBLASOrderRowMajor = 101
	CBLASOrderColMajor = 102
)

// enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
const (
	CBLASNoTrans   = 111
	CBLASTrans     = 112
	CBLASConjTrans = 113
)

// enum CBLAS_SIDE  {CblasLeft=141, CblasRight=142};
const (
	CBLASLeft  = 141
	CBLASRight = 142
)

// enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
const (
	CBLASUpper = 121
	CBLASLower = 122
)

// enum CBLAS_DIAG  {CblasNonUnit=131, CblasUnit=132};
const (
	CBLASDiagNonUnit = 131
	CBLASDiagUnit    = 132
)
