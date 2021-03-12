using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Network
    {
        static Random random = new Random(); 
        //Initialize parameters
        //Input
        static double x1 = 0.05;
        static double x2 = 0.07;
        //Target output
        static double z1 = 0.01;
        static double z2 = 0.99;
        //Weights
        static double w1 = random.NextDouble();
        static double w2 = random.NextDouble();
        static double w3 = random.NextDouble();
        static double w4 = random.NextDouble();
        static double w5 = random.NextDouble();
        static double w6 = random.NextDouble();
        static double w7 = random.NextDouble();
        static double w8 = random.NextDouble();
        //bias
        static double b1 = random.NextDouble();
        static double b2 = random.NextDouble();
        //training parameters        
        static double lr = 1.9; //learning rate
        
        //Forward propagation variables
        //hidden layer
        static double nety1;
        static double nety2;
        static double outy1;
        static double outy2;
        //output layer
        static double netz1;
        static double netz2;
        static double outz1;
        static double outz2;
        //Error variables
        static double Ez1;
        static double Ez2;
        public double Etotal;

        //weight gradient variables
        static double dw1;
        static double dw2;
        static double dw3;
        static double dw4;
        static double dw5;
        static double dw6;
        static double dw7;
        static double dw8;

        //forward propagation method
        public  void forward()
        {
            //hidden layer
            nety1 = w1 * x1 + w2 * x2 + b1 * 1;
            nety2 = w3 * x1 + w4 * x2 + b1 * 1;
            outy1 = Sigmoid(nety1);
            outy2 = Sigmoid(nety2);
            //output layer
            netz1 = w5 * outy1 + w6 * outy2 + b2 * 1;
            netz2 = w7 * outy1 + w8 * outy2 + b2 * 1;
            outz1 = Sigmoid(netz1);
            outz2 = Sigmoid(netz2);
            //Calculate total error
            Ez1 = 0.5 * Math.Pow((z1 - outz1), 2);
            Ez2 = 0.5 * Math.Pow((z2 - outz2), 2);
            Etotal = Ez1 + Ez2;
        }

        public void backward()
        {
            //gradient with respect to output layer outputs
            double doutz1 = outz1 - z1;
            double doutz2 = outz2 - z2;
            double dnetz1 = outz1 * (1 - outz1);
            double dnetz2 = outz2 * (1 - outz2);

            //gradient with respect to output layer weights
            dw5 = doutz1 * (outy1 * dnetz1);
            dw6 = doutz1 * (outy2 * dnetz1);
            dw7 = doutz2 * (outy1 * dnetz2);
            dw8 = doutz2 * (outy2 * dnetz2);

            //gradient with respect to hidden layer outputs
            double dEz1_outy1 = w5 * (doutz1 * dnetz1);
            double dEz2_outy1 = w7 * (doutz2 * dnetz2);
            double douty1 = dEz1_outy1 + dEz2_outy1;
            double dEz1_outy2 = w6 * (doutz1 * dnetz1);
            double dEz2_outy2 = w8 * (doutz2 * dnetz2);
            double douty2 = dEz1_outy2 + dEz2_outy2;
            double dnety1 = outy1 * (1 - outy1);
            double dnety2 = outy2 * (1 - outy2);

            //gradient with respect to hidden layer weights
            dw1 = douty1 * (x1 * dnety1);
            dw2 = douty1 * (x2 * dnety1);
            dw3 = douty2 * (x1 * dnety2);
            dw4 = douty2 * (x2 * dnety2);
        }

        public void weight_adjustment()
        {
            w1 = w1 - dw1 * lr;
            w2 = w2 - dw2 * lr;
            w3 = w3 - dw3 * lr;
            w4 = w4 - dw4 * lr;

            w5 = w5 - dw5 * lr;
            w6 = w6 - dw6 * lr;
            w7 = w7 - dw7 * lr;
            w8 = w8 - dw8 * lr;
        }

        public static double Sigmoid(double x)
        {
            return 1 / (1 + (Math.Exp(-x)));
        }


    }
}
