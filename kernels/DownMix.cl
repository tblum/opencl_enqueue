#define DOWNMIX_LEFT_SPEAKER_FRONT_LEFT	    1.0f
#define DOWNMIX_LEFT_SPEAKER_FRONT_RIGHT	0.0f
#define DOWNMIX_LEFT_SPEAKER_FRONT_CENTER	0.5f
#define DOWNMIX_LEFT_SPEAKER_LOW_FREQUENCY	0.707f
#define DOWNMIX_LEFT_SPEAKER_SIDE_LEFT	    0.707f
#define DOWNMIX_LEFT_SPEAKER_SIDE_RIGHT	    0.0f
#define DOWNMIX_LEFT_SPEAKER_BACK_LEFT	    0.707f
#define DOWNMIX_LEFT_SPEAKER_BACK_RIGHT	    0.0f
#define DOWNMIX_RIGHT_SPEAKER_FRONT_LEFT	0.0f
#define DOWNMIX_RIGHT_SPEAKER_FRONT_RIGHT	1.0f
#define DOWNMIX_RIGHT_SPEAKER_FRONT_CENTER	0.5f
#define DOWNMIX_RIGHT_SPEAKER_LOW_FREQUENCY	0.707f
#define DOWNMIX_RIGHT_SPEAKER_SIDE_LEFT	    0.0f
#define DOWNMIX_RIGHT_SPEAKER_SIDE_RIGHT	0.707f
#define DOWNMIX_RIGHT_SPEAKER_BACK_LEFT	    0.0f
#define DOWNMIX_RIGHT_SPEAKER_BACK_RIGHT	0.707f

#define IN_CH 8
#define OUT_CH 2

__kernel void downmix(__global const float *in, __global float *out, unsigned int frames)
{
    unsigned int i = get_global_id(0);
    if (i >= frames)
        return;
    float frontLeft    = in[i*IN_CH+0];
    float frontRight   = in[i*IN_CH+1];
    float frontCenter  = in[i*IN_CH+2];
    float lowFrequency = in[i*IN_CH+3];
    float backLeft     = in[i*IN_CH+4];
    float backRight    = in[i*IN_CH+5];
    float sideLeft     = in[i*IN_CH+6];
    float sideRight    = in[i*IN_CH+7];


    out[i*OUT_CH+0] = frontLeft    * DOWNMIX_LEFT_SPEAKER_FRONT_LEFT +
                      frontRight   * DOWNMIX_LEFT_SPEAKER_FRONT_RIGHT +
                      frontCenter  * DOWNMIX_LEFT_SPEAKER_FRONT_CENTER +
                      lowFrequency * DOWNMIX_LEFT_SPEAKER_LOW_FREQUENCY +
                      backLeft     * DOWNMIX_LEFT_SPEAKER_BACK_LEFT +
                      backRight    * DOWNMIX_LEFT_SPEAKER_BACK_RIGHT +
                      sideLeft     * DOWNMIX_LEFT_SPEAKER_SIDE_LEFT +
                      sideRight    * DOWNMIX_LEFT_SPEAKER_SIDE_RIGHT ;
    out[i*OUT_CH+1] = frontLeft    * DOWNMIX_RIGHT_SPEAKER_FRONT_LEFT +
                      frontRight   * DOWNMIX_RIGHT_SPEAKER_FRONT_RIGHT +
                      frontCenter  * DOWNMIX_RIGHT_SPEAKER_FRONT_CENTER +
                      lowFrequency * DOWNMIX_RIGHT_SPEAKER_LOW_FREQUENCY +
                      backLeft     * DOWNMIX_RIGHT_SPEAKER_BACK_LEFT +
                      backRight    * DOWNMIX_RIGHT_SPEAKER_BACK_RIGHT +
                      sideLeft     * DOWNMIX_RIGHT_SPEAKER_SIDE_LEFT +
                      sideRight    * DOWNMIX_RIGHT_SPEAKER_SIDE_RIGHT ;

}
