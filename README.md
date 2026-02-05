<p align='center'><a href='https://www.eventbrite.com/e/machine-learning-and-generative-ai-system-design-workshop-tickets-1975103644168?aff=Github'><img src='https://static.packt-cdn.com/assets/images/packt+events/Sairam_ML_GenAI_Github_banner.png'/></a></p>




# Machine Learning with Amazon SageMaker Cookbook 

<a href="https://www.packtpub.com/product/machine-learning-with-amazon-sagemaker-cookbook/9781800567030"><img src="https://static.packt-cdn.com/products/9781800567030/cover/smaller" alt="Book Name" height="256px" align="right"></a>

This is the code repository for [Machine Learning with Amazon SageMaker Cookbook](https://www.packtpub.com/product/machine-learning-with-amazon-sagemaker-cookbook/9781800567030), published by Packt.

**80 proven recipes for data scientists and developers to perform machine learning experiments and deployments**

## What is this book about?
Amazon SageMaker is a fully managed machine learning (ML) service that helps data scientists and ML practitioners manage ML experiments. In this book, you'll use the different capabilities and features of Amazon SageMaker to solve relevant data science and ML problems.

This book covers the following exciting features: 
* Train and deploy NLP, time series forecasting, and computer vision models to solve different business problems
* Push the limits of customization in SageMaker using custom container images
* Use AutoML capabilities with SageMaker Autopilot to create high-quality models
* Work with effective data analysis and preparation techniques
* Explore solutions for debugging and managing ML experiments and deployments
* Deal with bias detection and ML explainability requirements using SageMaker Clarify

If you feel this book is for you, get your [copy](https://www.amazon.com/Machine-Learning-Amazon-SageMaker-Cookbook/dp/1800567030) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter05.

The code will look like the following:

```
estimator = sagemaker.estimator.Estimator( 
    role=role_arn,
    instance_count=1,
    instance_type='ml.m5.2xlarge',
    image_uri=container,
    debugger_hook_config=debugger_hook_config,
    rules=rules,
    sagemaker_session=session)

```

**Following is what you need for this book:**
This book is for developers, data scientists, and machine learning practitioners interested in using Amazon SageMaker to build, analyze, and deploy machine learning models with 80 step-by-step recipes. All you need is an AWS account to get things running. Prior knowledge of AWS, machine learning, and the Python programming language will help you to grasp the concepts covered in this book more effectively.

With the following software and hardware list you can run all code files present in the book (Chapter 1-9).

### Software and Hardware List

| Chapter  | Software required                | OS required                        |
| -------- | ---------------------------------| -----------------------------------|
| 1-9      | AWS Account                      | Windows, Mac OS X, and Linux (Any) |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781800567030_ColorImages.pdf).

## Errata & Troubleshooting Tips

### `Page 76` (**Launching and preparing the Cloud9 environment**): 

In some cases, a **Cloud9** instance fails to launch due to **VPC** network configuration issues. If you see an error similar to `Unable to access your environment... failed to create: [Instance]...`, you may need to do one or more of the following to troubleshoot and solve the issue:

1. Use a different availability zone (e.g., `us-east-1c`)
2. Use the default VPC when launching the Cloud9 instance. If there is no default VPC, creating a new one with only public subnets would help get things working easily. You may use the **VPC Wizard** and choose the `VPC with a Single Public Subnet` option. Once this new VPC has been created, use this VPC along with the public subnet when configuring and creating a new Cloud9 instance.
3. Check if resources in the subnet selected (e.g., `subnet-abcdef | default in us-east-1a`) have internet access. This can be checked in the routing table configuration in the VPC console. Look for the route table where the subnet is associated (implicitly or explicitly) and check if we have this configuration: `[Destination] 0.0.0.0/0 and [Target] igw-abcdef`. Link: `https://console.aws.amazon.com/vpc/home?region=us-east-1#RouteTables`:
4. If none of the above works, use a different region with an existing default VPC and try different subnets.

For more information, [click here](https://docs.aws.amazon.com/cloud9/latest/user-guide/troubleshooting.html)

## Code in Action

Click on the following link to see the Code in Action:

https://bit.ly/3DYHjoB

### Related products <Other books you may enjoy>
* Learn Amazon SageMaker [[Packt]](https://www.packtpub.com/product/learn-amazon-sagemaker/9781800208919) [[Amazon]](https://www.amazon.in/Learn-Amazon-SageMaker-developers-scientists/dp/180020891X)

* Amazon SageMaker Best Practices [[Packt]](https://www.packtpub.com/product/amazon-sagemaker-best-practices/9781801070522) [[Amazon]](https://www.amazon.in/Amazon-SageMaker-Best-Practices-successful/dp/1801070520)

## Get to Know the Author
**Joshua Arvin Lat**
He is the Chief Technology Officer (CTO) of NuWorks Interactive Labs, Inc. He previously served as the CTO of three Australian-owned companies and also served as the director for software development and engineering for multiple e-commerce start-ups in the past, which allowed him to be more effective as a leader. Years ago, he and his team won first place in a global cybersecurity competition with their published research paper. He is also an AWS Machine Learning Hero and has shared his knowledge at several international conferences, discussing practical strategies on machine learning, engineering, security, and management.

### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781800567030">https://packt.link/free-ebook/9781800567030 </a> </p>