package tokenizer

type Token string

type Tokenizer struct {
	tokens     []Token
	tokenCodes map[Token]int
}

func (t *Tokenizer) Init(content string) {

}
