from evaluator import FaceShapeEvaluator

tester = FaceShapeEvaluator('/root/servers/server_1/model_ver2.h5')
result = tester.evaluate('/root/servers/server_1/round_test.jpg')
print(result)