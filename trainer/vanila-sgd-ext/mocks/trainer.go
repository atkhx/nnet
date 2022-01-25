// Code generated by MockGen. DO NOT EDIT.
// Source: trainer.go

// Package mocks is a generated GoMock package.
package mocks

import (
	reflect "reflect"

	data "github.com/atkhx/nnet/data"
	nnet "github.com/atkhx/nnet/layer"
	gomock "github.com/golang/mock/gomock"
)

// MockNet is a mock of Net interface
type MockNet struct {
	ctrl     *gomock.Controller
	recorder *MockNetMockRecorder
}

// MockNetMockRecorder is the mock recorder for MockNet
type MockNetMockRecorder struct {
	mock *MockNet
}

// NewMockNet creates a new mock instance
func NewMockNet(ctrl *gomock.Controller) *MockNet {
	mock := &MockNet{ctrl: ctrl}
	mock.recorder = &MockNetMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use
func (m *MockNet) EXPECT() *MockNetMockRecorder {
	return m.recorder
}

// Activate mocks base method
func (m *MockNet) Activate(inputs *data.Data) *data.Data {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Activate", inputs)
	ret0, _ := ret[0].(*data.Data)
	return ret0
}

// Activate indicates an expected call of Activate
func (mr *MockNetMockRecorder) Activate(inputs interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Activate", reflect.TypeOf((*MockNet)(nil).Activate), inputs)
}

// Backprop mocks base method
func (m *MockNet) Backprop(deltas *data.Data) *data.Data {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Backprop", deltas)
	ret0, _ := ret[0].(*data.Data)
	return ret0
}

// Backprop indicates an expected call of Backprop
func (mr *MockNetMockRecorder) Backprop(deltas interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Backprop", reflect.TypeOf((*MockNet)(nil).Backprop), deltas)
}

// GetLayersCount mocks base method
func (m *MockNet) GetLayersCount() int {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetLayersCount")
	ret0, _ := ret[0].(int)
	return ret0
}

// GetLayersCount indicates an expected call of GetLayersCount
func (mr *MockNetMockRecorder) GetLayersCount() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetLayersCount", reflect.TypeOf((*MockNet)(nil).GetLayersCount))
}

// GetLayer mocks base method
func (m *MockNet) GetLayer(index int) nnet.Layer {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetLayer", index)
	ret0, _ := ret[0].(nnet.Layer)
	return ret0
}

// GetLayer indicates an expected call of GetLayer
func (mr *MockNetMockRecorder) GetLayer(index interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetLayer", reflect.TypeOf((*MockNet)(nil).GetLayer), index)
}

// MockLoss is a mock of Loss interface
type MockLoss struct {
	ctrl     *gomock.Controller
	recorder *MockLossMockRecorder
}

// MockLossMockRecorder is the mock recorder for MockLoss
type MockLossMockRecorder struct {
	mock *MockLoss
}

// NewMockLoss creates a new mock instance
func NewMockLoss(ctrl *gomock.Controller) *MockLoss {
	mock := &MockLoss{ctrl: ctrl}
	mock.recorder = &MockLossMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use
func (m *MockLoss) EXPECT() *MockLossMockRecorder {
	return m.recorder
}

// GetDeltas mocks base method
func (m *MockLoss) GetDeltas(target, output *data.Data) *data.Data {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetDeltas", target, output)
	ret0, _ := ret[0].(*data.Data)
	return ret0
}

// GetDeltas indicates an expected call of GetDeltas
func (mr *MockLossMockRecorder) GetDeltas(target, output interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetDeltas", reflect.TypeOf((*MockLoss)(nil).GetDeltas), target, output)
}

// MockTrainableLayer is a mock of TrainableLayer interface
type MockTrainableLayer struct {
	ctrl     *gomock.Controller
	recorder *MockTrainableLayerMockRecorder
}

// MockTrainableLayerMockRecorder is the mock recorder for MockTrainableLayer
type MockTrainableLayerMockRecorder struct {
	mock *MockTrainableLayer
}

// NewMockTrainableLayer creates a new mock instance
func NewMockTrainableLayer(ctrl *gomock.Controller) *MockTrainableLayer {
	mock := &MockTrainableLayer{ctrl: ctrl}
	mock.recorder = &MockTrainableLayerMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use
func (m *MockTrainableLayer) EXPECT() *MockTrainableLayerMockRecorder {
	return m.recorder
}

// InitDataSizes mocks base method
func (m *MockTrainableLayer) InitDataSizes(w, h, d int) (int, int, int) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "InitDataSizes", w, h, d)
	ret0, _ := ret[0].(int)
	ret1, _ := ret[1].(int)
	ret2, _ := ret[2].(int)
	return ret0, ret1, ret2
}

// InitDataSizes indicates an expected call of InitDataSizes
func (mr *MockTrainableLayerMockRecorder) InitDataSizes(w, h, d interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "InitDataSizes", reflect.TypeOf((*MockTrainableLayer)(nil).InitDataSizes), w, h, d)
}

// Activate mocks base method
func (m *MockTrainableLayer) Activate(inputs *data.Data) *data.Data {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Activate", inputs)
	ret0, _ := ret[0].(*data.Data)
	return ret0
}

// Activate indicates an expected call of Activate
func (mr *MockTrainableLayerMockRecorder) Activate(inputs interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Activate", reflect.TypeOf((*MockTrainableLayer)(nil).Activate), inputs)
}

// Backprop mocks base method
func (m *MockTrainableLayer) Backprop(deltas *data.Data) *data.Data {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Backprop", deltas)
	ret0, _ := ret[0].(*data.Data)
	return ret0
}

// Backprop indicates an expected call of Backprop
func (mr *MockTrainableLayerMockRecorder) Backprop(deltas interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Backprop", reflect.TypeOf((*MockTrainableLayer)(nil).Backprop), deltas)
}

// GetWeightsWithGradient mocks base method
func (m *MockTrainableLayer) GetWeightsWithGradient() (*data.Data, *data.Data) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetWeightsWithGradient")
	ret0, _ := ret[0].(*data.Data)
	ret1, _ := ret[1].(*data.Data)
	return ret0, ret1
}

// GetWeightsWithGradient indicates an expected call of GetWeightsWithGradient
func (mr *MockTrainableLayerMockRecorder) GetWeightsWithGradient() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetWeightsWithGradient", reflect.TypeOf((*MockTrainableLayer)(nil).GetWeightsWithGradient))
}

// GetBiasesWithGradient mocks base method
func (m *MockTrainableLayer) GetBiasesWithGradient() (*data.Data, *data.Data) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetBiasesWithGradient")
	ret0, _ := ret[0].(*data.Data)
	ret1, _ := ret[1].(*data.Data)
	return ret0, ret1
}

// GetBiasesWithGradient indicates an expected call of GetBiasesWithGradient
func (mr *MockTrainableLayerMockRecorder) GetBiasesWithGradient() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetBiasesWithGradient", reflect.TypeOf((*MockTrainableLayer)(nil).GetBiasesWithGradient))
}
