using AI.ML.ANN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI.ML.Core
{
    public class Vector
    {
        public static double[] Logt(double[] x, params double[] z)
        {
            int r = x.Length;

            double u;
            double[] y = new double[r];

            for (int i = 0; i < r; i++)
            {
                u = 1.0 + System.Math.Exp(-1.0 * z[1] * x[i]);
                y[i] = z[0] / u;
            }

            return y;
        }

        public static void Logt(double[] _source, double[] _target, params double[] _logt)
        {
            int r = _source.Length;

            double u;

            for (int i = 0; i < r; i++)
            {
                u = 1.0 + System.Math.Exp(-1.0 * _logt[1] * _source[i]);
                _target[i] = _logt[0] / u;
            }
        }

        /// <summary>
        /// element wise multiplication for vectors
        /// </summary>
        /// <param name="x">vector</param>
        /// <param name="y">vector</param>
        /// <returns></returns>
        public static double[] MultiplyByElements(double[] x, double[] y)
        {
            if (x.Length != y.Length)
                throw new Exception("Size mismatch");

            int r = x.Length;

            double[] z = new double[r];

            for (int i = 0; i < r; i++)
                z[i] = x[i] * y[i];

            return z;
        }

        public static double[] MultiplyByElement(double[] x, double y)
        {
           
            int r = x.Length;

            double[] z = new double[r];

            for (int i = 0; i < r; i++)
                z[i] = x[i] * y;

            return z;
        }


        /// <summary>
        /// fill with random numbers with normal distribution
        /// </summary>
        /// <param name="x">vector</param>
        /// <param name="m">mean</param>
        /// <param name="v">variance</param>
        public static void RandomNormal(double[] x, double m, double v)
        {
            int r = x.Length;
            //
            for (int i = 0; i < r; i++)
                x[i] = Math.Daemon.NextGaussianDouble(m, v)[0];
        }

        public static void RandomStandardNormal(double[] x, double scale)
        {
            int r = x.Length;
            Random rand = new Random();
            for (int i = 0; i < r; i++)
            {
                double u1 = 1.0 - rand.NextDouble();
                double u2 = 1.0 - rand.NextDouble();
                double randStdNormal = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Sin(2.0 * System.Math.PI * u2); //random normal(0,1)
                x[i] = randStdNormal * scale;
            }                 
        }

        public static double[] MeanVariance(double[] x)
        {
            int r = x.Length;
            double sum = 0.0;
            for (int i = 0; i < r; i++)
               sum += x[i];               
            double mean = sum / r;
            
            sum = 0.0;
            for (int i = 0; i < r; i++)
                sum += System.Math.Pow(x[i] - mean, 2);
            double variance = sum / r;
            return new double[] { mean, variance };
        }

        public static void RandomUniform(double[] x, double min, double max)
        {
            int r = x.Length;
            double d = max - min;
            for (int i = 0; i < r; i++)
                x[i] = min + Math.Daemon.Random.NextDouble() * d;
        }
        
        /// <summary>
        /// tanh activation function
        /// </summary>
        /// <param name="_source">source</param>
        /// <param name="_tanh">function parameters</param>
        /// <returns></returns>
        public static double[] Tanh(double[] _source, params double[] _tanh)
        {
            int r = _source.Length;

            double u; double[] y = new double[r];

            for (int i = 0; i < r; i++)
            {
                u = _tanh[1] * _source[i];
                y[i] = _tanh[0] * System.Math.Tanh(u);
            }

            return y;
        }

        /// <summary>
        /// tanh activation function
        /// </summary>
        /// <param name="_source">source</param>
        /// <param name="_output">target</param>
        /// <param name="_tanh">function parameters</param>
        public static void Tanh(double[] _source, double[] _output, params double[] _tanh)
        {
            int r = _source.Length;

            double u;

            for (int i = 0; i < r; i++)
            {
                u = _tanh[1] * _source[i];
                _output[i] = _tanh[0] * System.Math.Tanh(u);
            }
        }

        /// <summary>
        /// print vector
        /// </summary>
        /// <param name="x">vector</param>
        /// <returns></returns>
        public static string ToString(double[] x)
        {
            string s = ""; char[] t = new char[] { ' ' };

            int r = x.Length;

            for (int i = 0; i < r; i++)
            {
                s += "\n[";
                s += Math.Method.Sign(x[i]) + x[i].ToString(Global.Precision) + " ";
                s = s.TrimEnd(t) + "]";
            }

            return s;
        }

        public static double[] Zero(int l)
        {
            double[] z = new double[l];
            for (int i = 0; i < l; i++)
                z[i] = 0.0;
            return z;
        }

        public static void Zero(double[] z)
        {
            for (int i = 0; i < z.Length; i++)
                z[i] = 0.0;
        }

        public static double[] Constant(double x, int length)
        {
            double[] a = new double[length];

            for (int i = 0; i < length; i++)
                a[i] = x;

            return a;
        }

        /// <summary>
        /// make copy of vector
        /// </summary>
        /// <param name="m">vector</param>
        /// <returns>copy of vector</returns>
        public static double[] Copy(double[] m)
        {
            int r = m.Length;
            double[] x = new double[r];
            for (int i = 0; i < r; i++)
                x[i] = m[i];
            return x;
        }

        public static double[] Copy(double[] matrix, int l, int length)
        {
            double[] m = new double[length];
            for (int i = 0; i < length; i++)
                m[i] = matrix[l + i];
            return m;
        }       

        public static void InsertAt(double[] x, double[] y, int offset)
        {
            int r = x.Length;
            for (int i = 0; i < r; i++)
                y[i + offset] = x[i];
        }

        public static void AddAt(double[] x, double[] y, int offset)
        {
            int r = x.Length;
            for (int i = 0; i < r; i++)
                y[i + offset] += x[i];
        }

        public static void InsertAt(double[] x, double[] y, int xr, int xc, int u, int v)
        {
            for (int i = 0; i < xr; i++)
                for (int j = 0; j < xc; j++)
                    x[(xc * (i + u)) + (j + v)] += y[(xc * i) + j];
        }

        public static void AddAt(double[] x, double[] y, int xr, int yc, int u, int v)
        {
            for (int idx = 0, i = 0; i < yc; i++)
                for (int j = 0; j < yc; j++)
                    x[(xr * (i + u)) + (j + v)] += y[idx++];
        }

        internal static double[] Slice(double[] x, int xr, int xc, int u, int v, int f)
        {
            double[] s = new double[f * f];
            var index_y = u;
            int c = 0;
            for (int i = 0; i < f; i++)
            {
                var index_x = v;
                for (int j = 0; j < f; j++)
                {
                    s[c++] = x[(xc * (index_y)) + index_x];
                    //slice.value[i][j] = src.value[index_y][index_x];
                    index_x++;
                }
                index_y++;
            }            
            return s;
        }

        public static double[] Transpose(double[] x, int xr, int xc)
        {
            double[][] matrix = ToMatrix(x, xr, xc);
            matrix = Matrix.Transpose(matrix);
            double[] t = Matrix.Flatten(matrix);            
            return t;            
        }

        public static double[][] ToMatrix(double[] x, int xr, int xc)
        {
            double[][] t = new double[xr][];
            for (int i = 0; i < xr; i++)
                t[i] = new double[xc]; 

            for (int i = 0; i < xr; i++)
                for (int j = 0; j < xc; j++)
                    t[i][j] = x[(xc * i) + j];
            return t;
        }

        public static void AddAt(double[] x, double[] y)
        {
            int r = x.Length;
            for (int i = 0; i < r; i++)
                y[i] += x[i];
        }

        public static double[] Subtract(double[] x, double[] y)
        {
            int r = x.Length;
            double[] s = new double[r];
            for (int i = 0; i < r; i++)
                s[i] = x[i] - y[i];
            return s;
        }

        public static double[] Add(double[] x, double[] y)
        {
            int r = x.Length;
            double[] s = new double[r];
            for (int i = 0; i < r; i++)
                s[i] = x[i] + y[i];
            return s;
        }

        public static void Add(double[] x, double y)
        {
            int r = x.Length;
            for (int i = 0; i < r; i++)
                x[i] += y;
        }

        public static double[] AddValue(double[] x, double y)
        {
            int r = x.Length;
            double[] z = new double[r];
            for (int i = 0; i < r; i++)
                z[i] = x[i] + y;
            return z;
        }

        public static void Add(double[] x, double[] y, double[,] _output)
        {
            int r = x.Length;
            for (int i = 0; i < r; i++)
                _output[i, 0] = x[i] + y[i];
        }
        public static void Add(double[] x, double[] y, double[] _output)
        {
            int r = x.Length;
            for (int i = 0; i < r; i++)
                _output[i] = x[i] + y[i];
        }

        public static double Sum(double[] x) 
        {
            double s = x.Sum();
            return s;
        }

        public static void InsertValue(int c, double[] x, int i, int j, double value)
        {
            x[(c * i) + j] = value;
        }

        public static double GetValueAt(int c, double[] x, int i, int j)
        {
            return x[(c * i) + j];
        }

        public static double[] Divide(double[] x, double value)
        {
            int r = x.Length;
            double[] y = new double[r];           

            for (int i = 0; i < r; i++)
                y[i] = x[i] / value;
            return y;
        }

        public static double[] Divide(double[] x, double[] z)
        {
            int r = x.Length;
            double[] y = new double[r];

            for (int i = 0; i < r; i++)
                y[i] = x[i]/z[i];
            return y;
        }

        public static double[] Multiply(double[] a, double b)
        {
            int r = a.Length;
            double[] y = new double[r];
            
            for (int i = 0; i < r; i++)
            {
                y[i] = a[i] * b;               
            }
            return y;
        }

        public static double[] Sqrt(double[] a)
        {
            int r = a.Length;
            double[] y = new double[r];
            
            for (int i = 0; i < r; i++)
            {
                y[i] = System.Math.Sqrt(a[i]);
            }
            return y;
        }

        public static double[] Pow(double[] a, int pow)
        {
            int r = a.Length;
            double[] y = new double[r];
            
            for (int i = 0; i < r; i++)
            {
                y[i] = System.Math.Pow(a[i], pow);
            }

            return y;
        }
    }
}
