<h1>WGANSing: A Multi-Voice Singing Voice Synthesizer Based on the Wasserstein-GAN</h1>

<h2>Pritish Chandna, Merlijn Blaauw, Jordi Bonada, Emilia Gómez</h2>

<h2>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for multi-voice singing voice synthesis
<h3>Installation</h3>
To install, clone the repository and use <pre><code>pip install -r requirements.txt </code></pre> to install the packages required.

 The main code is in the *main.py* file.  
 




<h3>Training and inference</h3>

To use the WGANSing, you will have to download the <a href="https://drive.google.com/file/d/1Mmg3cq5CYl-yOePHhiFWInizlr3tMPEE/view?usp=sharing" rel="nofollow"> model weights</a> and place it in the *log_dir* directory, defined in *config.py*. 

The NUS-48E dataset can be downloaded from <a href="https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/" rel="nofollow"> here</a>. Once downloaded, please change *wav_dir_nus* in *config.py* to the same directory that the dataset is in. 

To prepare the data for use, please use *prep_data_nus.py*.


Once setup, you can run the following commands. 
To train the model: 
<pre><code>python main.py -t</code></pre>. 
To synthesize a .lab file:
Use <pre><code>python main.py -e <i>filename</i> <i>alternate_singer_name</i> </code></pre> 

If no alternate singer is given then the original singer will be used for synthesis. A list of valid singer names will be displayed if an invalid singer is entered. 

You will also be prompted on wether plots showed be displayed or not, press *y* or *Y* to view plots.



<h2>Acknowledgments</h2>
The TITANX used for this research was donated by the NVIDIA Corporation. This work is partially supported by the Towards Richer Online Music Public-domain Archives <a href="https://trompamusic.eu/" rel="nofollow">(TROMPA)</a> (H2020 770376) European project.
          <p>[1] Duan, Zhiyan, et al. "The NUS sung and spoken lyrics corpus: A quantitative comparison of singing and speech." 2013 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference. IEEE, 2013.</p>
          <p>[2] Blaauw, Merlijn, and Jordi Bonada. "A Neural Parametric Singing Synthesizer Modeling Timbre and Expression from Natural Songs." Applied Sciences 7.12 (2017): 1313.</p>
          <p>[3] Blaauw, Merlijn, et al. “Data efficient voice cloning forneural  singing  synthesis,”  in2019  IEEE  International  Conference  onAcoustics, Speech and Signal Processing (ICASSP), 2019.</p>
