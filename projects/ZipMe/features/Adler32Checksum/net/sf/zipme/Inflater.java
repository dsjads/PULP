

package net.sf.zipme;
class Inflater {
  private Adler32 adler;

   public int getAdler(){
    return needsDictionary() ? readAdler : (int)adler.getValue();
  }
   protected void hook32(){
    this.adler=new Adler32();
    original();
  }

   public void end(){
    original();
    adler=null;
  }
   protected void hook33(  byte[] buf,  int off,  int more) throws DataFormatException {
    adler.update(buf,off,more);
    original(buf,off,more);
  }

   public void reset(){
    original();
    adler.reset();
  }
   protected void hook34(  byte[] buffer,  int off,  int len){
    adler.update(buffer,off,len);
    if ((int)adler.getValue() != readAdler)     throw new IllegalArgumentException("Wrong adler checksum");
    adler.reset();
    original(buffer,off,len);
  }
   protected void hook35() throws DataFormatException {
    if ((int)adler.getValue() != readAdler)     throw new DataFormatException("Adler chksum doesn't match: " + Integer.toHexString((int)adler.getValue()) + " vs. "+ Integer.toHexString(readAdler));
    original();
  }
}
