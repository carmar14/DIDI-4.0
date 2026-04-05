#include "weights.h"

//------valores del escaler-normalizar----
float mean[11] = {25.10691805, 23.99912999,  19.9577943,   54.85316511, 612.63726538,
   6.52410205,  30.10675088,  30.26448401,  30.11208846,  34.74959091
   0.98776368};
float std[11]  = {8.67410851,   3.44012685,   2.93085089,   8.78125467, 228.22370021,
   0.58151297,  11.50959722,  11.46206733,  11.66322201,   8.76334132,
   0.57487601};
//------valores del escaler-desnormalizar----
float mean_y[4] = {23.99912999,  19.9577943,  612.63726538,   6.52410205};
float std_y[4]  = {3.44012685,   2.93085089, 228.22370021,   0.58151297};

float relu(float x) {
  return x > 0 ? x : 0;
}

void predict(float input[11], float output[4]) {

  float h1[16];
  float h2[8];

  // Capa 1
  for (int j = 0; j < 16; j++) {
    h1[j] = b1[j];
    for (int i = 0; i < 11; i++) {
      h1[j] += input[i] * W1[i][j];
    }
    h1[j] = relu(h1[j]);
  }

  // Capa 2
  for (int j = 0; j < 8; j++) {
    h2[j] = b2[j];
    for (int i = 0; i < 16; i++) {
      h2[j] += h1[i] * W2[i][j];
    }
    h2[j] = relu(h2[j]);
  }

  // Capa de salida
  for (int j = 0; j < 4; j++) {
    output[j] = b3[j];
    for (int i = 0; i < 8; i++) {
      output[j] += h2[i] * W3[i][j];
    }
  }
}

void normalize(float input[11]) {
  for (int i = 0; i < 11; i++) {
    input[i] = (input[i] - mean[i]) / std[i];
  }
}

void denormalize(float output[4]) {
  for (int i = 0; i < 4; i++) {
    output[i] = output[i] * std_y[i] + mean_y[i];
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
}

void loop() {
  //------11 numero de sensores----
  float input[11] = { /* sensores */ }; 
  float output[4];

  normalize(input);

  predict(input, output);

  denormalize(output);

  Serial.println("Predicciones:");
  for (int i = 0; i < 4; i++) {
    Serial.println(output[i]);
  }

  delay(2000);

}
