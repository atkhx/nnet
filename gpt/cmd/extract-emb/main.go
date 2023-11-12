package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/atkhx/nnet/layer"
)

var (
	filename   = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/gpt/train_rms_2_config.json"
	embeddings = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/gpt/embeddings.json"
)

type Model struct {
	Layers layer.Layers
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	t := time.Now()
	config, err := os.ReadFile(filename)
	if err != nil && errors.Is(err, os.ErrNotExist) {
		err = fmt.Errorf("trained config not found (skip)")
		return
	}

	if err != nil {
		err = fmt.Errorf("read file failed: %w", err)
		return
	}
	fmt.Println("read config success:", time.Since(t))

	m := map[string]any{}

	if err = json.Unmarshal(config, &m); err != nil {
		err = fmt.Errorf("unmarshal config failed: %w", err)
		return
	}

	fmt.Println("unmarshal success:", time.Since(t))

	l := m["Layers"].([]any)

	m = map[string]any{
		"Layers": []any{l[0]},
	}

	os.Remove(embeddings)

	b, err := json.Marshal(m)
	if err != nil {
		err = fmt.Errorf("marshal embeddings: %w", err)
		return
	}

	err = os.WriteFile(embeddings, b, os.ModePerm)
}
