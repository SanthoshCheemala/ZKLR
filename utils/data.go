package utils

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

type Sample struct {
	Marks float64
	Label int
}

func LoadDataset(filename string) ([]Sample, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open dataset: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %w", err)
	}

	var samples []Sample
	for i, record := range records {
		if i == 0 {
			continue
		}

		marks, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid marks at line %d: %w", i+1, err)
		}

		label, err := strconv.Atoi(record[1])
		if err != nil {
			return nil, fmt.Errorf("invalid label at line %d: %w", i+1, err)
		}

		samples = append(samples, Sample{
			Marks: marks,
			Label: label,
		})
	}

	return samples, nil
}

func LoadModelParameters(filename string) (w, b float64, err error) {
	file, err := os.Open(filename)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	_, err = fmt.Fscanf(file, "W: %f\nB: %f", &w, &b)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to parse model parameters: %w", err)
	}

	return w, b, nil
}