package com.example.fingerprint_auth

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.fingerprint_auth.ml.Vgg
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var imgView: ImageView
    private lateinit var btnSelect: Button
    private lateinit var btnPredict: Button
    private lateinit var resView: TextView
    private lateinit var bitmap: Bitmap
    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnSelect = findViewById(R.id.btnSelect)
        btnPredict = findViewById(R.id.btnPredict)
        resView = findViewById(R.id.resView)
        imgView = findViewById(R.id.imgView)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(256, 300, ResizeOp.ResizeMethod.BILINEAR))
            .add(CastOp(DataType.FLOAT32))
            .add(NormalizeOp(0.0f, 255.0f))
            .build()

        btnSelect.setOnClickListener {
            resView.text = "Prediction"
            resView.setTextColor(Color.LTGRAY)
            val intent = Intent()
            intent.action = Intent.ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }

        btnPredict.setOnClickListener {
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)

            val model = Vgg.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 300, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)

            val maxIndex = outputs.outputFeature0AsTensorBuffer.floatArray[0]

            Log.d(ContentValues.TAG, "value of sigmoid is : $maxIndex")

            if(maxIndex < 0.5){
                resView.text = "Forged"
                resView.setTextColor(Color.RED)
            }else{
                resView.text = "Genuine"
                resView.setTextColor(Color.GREEN)
            }

            // Releases model resources if no longer used.
            model.close()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == 100){
            val uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imgView.setImageBitmap(bitmap)
        }
    }
}