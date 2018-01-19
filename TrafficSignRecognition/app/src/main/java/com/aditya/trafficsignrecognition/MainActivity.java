package com.aditya.trafficsignrecognition;

import android.app.Activity;
//PointF holds two float coordinates
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
//A mapping from String keys to various Parcelable values (interface for data container values, parcels)
import android.net.Uri;
import android.os.Bundle;

import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
//A user interface element the user can tap or click to perform an action.
import android.widget.Button;
//A user interface element that displays text to the user. To provide user-editable text, see EditText.
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
//Resizable-array implementation of the List interface. Implements all optional list operations, and permits all elements,
// including null. In addition to implementing the List interface, this class provides methods to
// //manipulate the size of the array that is used internally to store the list.
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
// basic list
import java.util.List;
//encapsulates a classified image
//public interface to the classification class, exposing a name and the recognize function
import com.aditya.trafficsignrecognition.model.Classification;
import com.aditya.trafficsignrecognition.model.Classifier;
//contains logic for reading labels, creating classifier, and classifying
import com.aditya.trafficsignrecognition.model.TensorFlowClassifier;
import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.INTER_NEAREST;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.resize;


public class MainActivity extends Activity {

    private static final int PIXEL_WIDTH = 32;

    // ui elements
    private Button pickBtn;
    private TextView resText;
    private ImageView imgview;

    private List<Classifier> mClassifiers = new ArrayList<>(); //list of classifiers, in our case 1

    private float[][] std = new float[32][32]; //training dataset standard deviation
    private float[][] avg = new float[32][32]; //training dataset mean

    Activity c;

    BaseLoaderCallback mBaseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            super.onManagerConnected(status);

        }
    };


    @Override
    // In the onCreate() method, you perform basic application startup logic that should happen
    //only once for the entire life of the activity.
    protected void onCreate(Bundle savedInstanceState) {
        //initialization
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        //get image view from XML to display the selected image
        imgview = (ImageView) findViewById(R.id.selected_image);

        //get button for initiating process
        pickBtn = (Button) findViewById(R.id.pick_image);

        // res text
        //this is the text that shows the output of the classification
        resText = (TextView) findViewById(R.id.result);

        c = this;

        // tensorflow
        //load up our saved model to perform inference from local storage
        loadModel();

        pickBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                // start picker to get image for cropping and then use the image in cropping activity
                CropImage.activity()
                        .setGuidelines(CropImageView.Guidelines.ON)
                        .start(c);

            }
        });

        read_normalization_data();
    }

    //the activity lifecycle


    //creates a model object in memory using the saved tensorflow protobuf model file
    //which contains all the learned weights
    private void loadModel() {
        //The Runnable interface is another way in which you can implement multi-threading other than extending the
        // //Thread class due to the fact that Java allows you to extend only one class. Runnable is just an interface,
        // //which provides the method run.
        // //Threads are implementations and use Runnable to call the method run().
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    //add 2 classifiers to our classifier arraylist
                    //the tensorflow classifier and the keras classifier
                    /*mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "TensorFlow",
                                    "opt_mnist_convnet-tf.pb", "labels.txt", PIXEL_WIDTH,
                                    "input", "output", true));*/
                    mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "Keras",
                                    "opt_gtsrb_convnet.pb", "labels.txt", PIXEL_WIDTH,
                                    "zero_padding2d_1_input", "dense_2/Softmax", false));
                } catch (final Exception e) {
                    //if they aren't found, throw an error!
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);
            if (resultCode == RESULT_OK) {
                Uri resultUri = result.getUri();
                try {
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), resultUri); //input image

                    Mat img = new Mat();

                    Utils.bitmapToMat(bitmap, img);

                    Mat grayscale = new Mat();
                    Mat resized = new Mat();

                    //converting to grayscale
                    cvtColor(img, grayscale, COLOR_RGB2GRAY);
                    //resizing the image to input size
                    resize(grayscale, resized, new Size(32, 32), 0.0, 0.0, INTER_NEAREST);
                    //converting integer values to floating point values
                    resized.convertTo(resized, CV_32F);

                    String ty = Integer.toString(resized.type());

                    Utils.matToBitmap(grayscale, bitmap);
                    imgview.setImageBitmap(bitmap);

                    //normalizing the input image
                    float[] val = new float[1];
                    for (int i = 0; i < 32; i++) {
                        for (int j = 0; j < 32; j++) {
                            resized.get(i, j, val);
                            val[0] = (val[0] - avg[i][j]) / std[i][j];
                            resized.put(i,j,val);
                        }
                    }

                    float[] arr = new float[ 32*32*1 ]; //input array

                    //preparing input for inference
                    for(int i=0;i<32;i++)
                    {
                        for(int j=0;j<32;j++)
                        {
                            resized.get(i, j, val);
                            arr[ (i*32 + j) + 0 ] = val[0];
                        }
                    }

                    for (Classifier classifier : mClassifiers) {
                        //perform classification on the image
                        final Classification res = classifier.recognize(arr);
                        //if it can't classify, output a question mark
                        if (res.getLabel() == null) {
                            resText.setText("?");
                        } else {
                            //else output its name
                            resText.setText(res.getLabel());
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
                Exception error = result.getError();
            }
        }
    }

    //reading the normalization data from the text files
    public void read_normalization_data() {
        InputStream is = null;

        String temp = "";
        char t;

        try {
            is = getAssets().open("avg.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            while (is.available() > 0) {
                for (int i = 0; i < 32; i++) {
                    for (int j = 0; j < 32; j++) {
                        temp = "";
                        t = 'a';
                        do {
                            t = (char) is.read();
                            if (t != 'e')
                                temp = temp + t;
                        } while (t != 'e');
                        avg[i][j] = Float.parseFloat(temp) * 10.0f;

                        is.read();
                        is.read();
                        is.read();
                        is.read();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Log.d("character", "avg: " + Float.toString(avg[31][31]));

        try {
            is = getAssets().open("std.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            while (is.available() > 0) {
                for (int i = 0; i < 32; i++) {
                    for (int j = 0; j < 32; j++) {
                        temp = "";
                        t = 'a';
                        do {
                            t = (char) is.read();
                            if (t != 'e')
                                temp = temp + t;
                        } while (t != 'e');
                        std[i][j] = Float.parseFloat(temp) * 10.0f;

                        is.read();
                        is.read();
                        is.read();
                        is.read();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Log.d("character", "std: " + Float.toString(std[31][31]));

    }

    @Override
    protected void onResume() {
        super.onResume();
        //Checking if opencv library has been loaded
        if (OpenCVLoader.initDebug()) {
            Log.d("OCV", "Open cv loaded");
            mBaseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d("OCV", "Open cv not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mBaseLoaderCallback);
        }
    }

};