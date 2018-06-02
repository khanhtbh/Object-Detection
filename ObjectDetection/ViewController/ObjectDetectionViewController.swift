//
//  ObjectDetectionViewController.swift
//  ObjectDetection
//
//  Created by Kei on 6/2/18.
//  Copyright © 2018 Kei. All rights reserved.
//

import Cocoa
import Vision
import AVFoundation

class ObjectDetectionViewController: ViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // video capture session
    let session = AVCaptureSession()
    // preview layer
    var previewLayer: AVCaptureVideoPreviewLayer!
    // queue for processing video frames
    let captureQueue = DispatchQueue(label: "captureQueue")
    // overlay layer
    var gradientLayer: CAGradientLayer!
    // vision request
    var visionRequests = [VNRequest]()
    
    private let visionSequenceHandler = VNSequenceRequestHandler()
    
    var recognitionThreshold : Float = 0.25

    @IBOutlet weak var previewView: NSView!
    
    @IBOutlet weak var resultText: NSTextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do view setup here.
        self.title = "Object Detection"
        initAvSession()
    }
    
    override func viewDidAppear() {
        super.viewDidAppear()
        
    }
    
    private func initAvSession() {
        guard let camera = AVCaptureDevice.default(for: .video) else {
            fatalError("No video camera available")
        }
        do {
            // add the preview layer
            previewLayer = AVCaptureVideoPreviewLayer(session: session)
            previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
            previewView.layer?.addSublayer(previewLayer)
            // add a slight gradient overlay so we can read the results easily
            gradientLayer = CAGradientLayer()
            gradientLayer.colors = [
                NSColor.init(red: 0, green: 0, blue: 0, alpha: 0.7).cgColor,
                NSColor.init(red: 0, green: 0, blue: 0, alpha: 0.0).cgColor,
            ]
            gradientLayer.locations = [0.0, 0.3]
            self.previewView.layer?.addSublayer(gradientLayer)
            
            // create the capture input and the video output
            let cameraInput = try AVCaptureDeviceInput(device: camera)
            
            let videoOutput = AVCaptureVideoDataOutput()
            videoOutput.setSampleBufferDelegate(self, queue: captureQueue)
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            session.sessionPreset = .high
            
            // wire up the session
            session.addInput(cameraInput)
            session.addOutput(videoOutput)
            
            // make sure we are in portrait mode
            let conn = videoOutput.connection(with: .video)
            conn?.videoOrientation = .portrait
            
            // Start the session
            session.startRunning()
            
            // set up the vision model
            guard let resNet50Model = try? VNCoreMLModel(for: Resnet50().model) else {
                fatalError("Could not load model")
            }
            // set up the request using our vision model
            let classificationRequest = VNCoreMLRequest(model: resNet50Model, completionHandler: handleClassifications)
            classificationRequest.imageCropAndScaleOption = .centerCrop
            visionRequests = [classificationRequest]
        } catch {
            fatalError(error.localizedDescription)
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        connection.videoOrientation = .portrait
        
        var requestOptions:[VNImageOption: Any] = [:]
        
        if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCGImagePropertyPNGDictionary, nil) {
            requestOptions = [.cameraIntrinsics: cameraIntrinsicData]
        }
        
        // for orientation see kCGImagePropertyOrientation
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .upMirrored, options: requestOptions)
        do {
            try imageRequestHandler.perform(self.visionRequests)
        } catch {
            print(error)
        }
    }
    func handleClassifications(request: VNRequest, error: Error?) {
        if let theError = error {
            print("Error: \(theError.localizedDescription)")
            return
        }
        guard let observations = request.results else {
            print("No results")
            return
        }
        
        let classifications = observations[0...4] // top 4 results
            .compactMap({ $0 as? VNClassificationObservation })
            .compactMap({$0.confidence > recognitionThreshold ? $0 : nil})
            .map({ "\($0.identifier) - accurate at \(String(format:"%.2f%%", $0.confidence * 100.0))" })
            .joined(separator: "\n")
        
        DispatchQueue.main.async {
            self.resultText.stringValue = classifications
        }
        
    }
    
    override func viewDidLayout() {
        super.viewDidLayout()
        previewLayer.frame = previewView.bounds
    }
    
    override func viewWillDisappear() {
        session.stopRunning()
    }
    
    deinit {
        
    }
}
