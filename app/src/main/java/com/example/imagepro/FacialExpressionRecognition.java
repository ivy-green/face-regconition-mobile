package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class FacialExpressionRecognition {
    // define interpreter
    private Interpreter interpreter;
    //
    private int INPUT_SIZE;
    // define height and width of original frame
    private int height = 0;
    private int width = 0;
    // define GPU delegate, if using GPU
//    private GpuDelegate gpuDelegate = null

    // define cascadeClassifier for face detection
    private CascadeClassifier cascadeClassifier;

    //
    FacialExpressionRecognition(AssetManager assetManager, Context context, String modelPath, int inputSize) throws IOException {
        INPUT_SIZE = inputSize;
        // set GPU for the interpreter
        Interpreter.Options options = new Interpreter.Options();
//        gpuDelegate = new GpuDelegate();
        // add gpu Delegate to interpreter
//        options.addDelegate(gpuDelegate);

        options.setNumThreads(4); // depend on device
        // load model weight to interpreter
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        // if model is load print
        Log.d("Facial_Expression", "Model is load");

        // load haarcascade classifier
        try {
            // define inpput stream to read classifier
            InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            // create folder
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            // create file in that folder
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt");
            // define output stream to transfer data to file we created
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            // create buffer to store byte
            byte[] buffer = new byte[4096];
            int byteRead;
            // read byte in while load loop
            // when it read -1 that mean no data to read
            while((byteRead = is.read(buffer)) != -1) {
                // writing byteRead to buffer
                os.write(buffer, 0, byteRead);
            }

            // close is and os
            is.close();
            os.close();
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            // if cascade file is loaded print
            Log.d("Facial_Expression", "Classifier is loaded");

        } catch (IOException e){
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // give description of file
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelPath);
        // create a inputstream to read file
        FileInputStream inputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // call in onCameraFrame
    public Mat recognizeImage(Mat mat_image){
        // before predicting
        // image is not properly align
        // we have to rotate it by 90 degree for proper prediction
        Core.flip(mat_image.t(), mat_image, 1); // rotate by 90 degree

        // start with process
        // convert mat_image to gray  scale
        Mat grayScaleImage = new Mat();
        Imgproc.cvtColor(mat_image, grayScaleImage, Imgproc.COLOR_BGR2GRAY);

        // set height and width
        height = grayScaleImage.height();
        width = grayScaleImage.width();

        // define minimum height of face in original image
        // below this size no face in original image will show
        int absoluteFaceSize = (int) (height * 0.1);
        // create MatOfRect to store face
        MatOfRect faces = new MatOfRect();
        // check if cascadeClassifier is loaded or not
        if(cascadeClassifier != null) {
            // detect face in frame
            cascadeClassifier.detectMultiScale(grayScaleImage, // input
                    faces, // output
                    1.1, 2, 1,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size()); // minimum size


            // convert it to array
            Rect[] faceArray = faces.toArray();
            // loop through each face
            for(int i = 0; i < faceArray.length; i++){
                // draw rectangle to face
                Imgproc.rectangle(
                        mat_image, // input/ output
                        faceArray[i].tl(), // starting point
                        faceArray[i].br(), // ending point
                        new Scalar(0,255,0,255), 2); // R G B alpha

                // crop face from original frame and grayScaleImage
                Rect roi = new Rect((int) faceArray[i].tl().x, // starting point
                        (int) faceArray[i].tl().y, // ending point
                        ((int) faceArray[i].br().x) - ((int) faceArray[i].tl().x), // width
                        ((int) faceArray[i].br().y) - ((int) faceArray[i].tl().y)); // height

                Mat cropped_rgba = new Mat(mat_image, roi);
                // convert cropped_rgba to bitmap
                Bitmap bitmap = null;
                bitmap = Bitmap.createBitmap(cropped_rgba.cols(), cropped_rgba.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped_rgba, bitmap);
                // resize bitmap to (48,48)
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 48, 48, false);
                // now convert scaledBitmap to byteBuffer
                ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

                // create an obj to hold output
                float[][] expression = new float[1][1];
                // predict with byteBuffer as an input and emotion as an output
                interpreter.run(byteBuffer, expression);

                // define float value of expression
                float express_v = (float) Array.get(Array.get(expression, 0), 0);
                // create function to return text expression
                String expression_s = get_emotion_text(express_v);
                Log.d("Facial_Expression", "Output: " + expression_s + " - " + express_v);

                Imgproc.putText(mat_image,  expression_s + " (" + express_v + ")",
                        new Point(faceArray[i].tl().x + 1, faceArray[i].tl().y + 20),
                        1, 1.5, new Scalar(0,0,255,150), 2);
            }
        }
        // after predicting
        // rotate mat_image back
        Core.flip(mat_image.t(), mat_image, 0); // rotate by 90 degree

        return mat_image;
    }

    private String get_emotion_text(float expressV) {
        // create empty string
        String val = "";
        // use if statement to detect
        if(expressV >= 0 && expressV < 0.5) {
            val = "angry";
        }
        else if(expressV >= 0.5 && expressV < 1.5) {
            val = "disgust";
        }
        else if(expressV >= 1.5 && expressV < 2.5) {
            val = "fear";
        }
        else if(expressV >= 2.5 && expressV < 3.5) {
            val = "happy";
        }
        else if(expressV >= 3.5 && expressV < 4.5) {
            val = "neutral";
        }
        else if(expressV >= 4.5 && expressV < 5.5) {
            val = "sad";
        }
        else{
            val = "surprise";
        }

        return val;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        int size_image = INPUT_SIZE; // 48
        byteBuffer = ByteBuffer.allocateDirect(4*1*size_image*size_image*3);
        // 4: multiplied for float input
        // 3: multiplied for rgb
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_image * size_image];
        scaledBitmap.getPixels(intValues,
                0,
                scaledBitmap.getWidth(),
                0, 0,
                scaledBitmap.getWidth(),
                scaledBitmap.getHeight());
        int pixel = 0;
        for(int i = 0; i < size_image; ++i){
            for (int j = 0; j < size_image; ++j){
                final int val = intValues[pixel++];
                // scale image to convert image from 0-255 to 0-1
                byteBuffer.putFloat((((val>>16) & 0xFF) / 255.0f));
                byteBuffer.putFloat((((val>>8) & 0xFF) / 255.0f));
                byteBuffer.putFloat((val & 0xFF) / 255.0f);

            }
        }

        return byteBuffer;
    }
}
