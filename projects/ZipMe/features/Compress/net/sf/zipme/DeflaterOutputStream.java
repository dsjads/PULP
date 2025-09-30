

package net.sf.zipme;
class DeflaterOutputStream {

   protected void deflate() throws IOException {
    while (!def.needsInput()) {
      int len=def.deflate(buf,0,buf.length);
      if (len <= 0)       break;
      out.write(buf,0,len);
    }
    if (!def.needsInput())     throw new Error("Can't deflate all input?");
  }

   public void flush() throws IOException {
    def.flush();
    deflate();
    out.flush();
  }

   public void finish() throws IOException {
    def.finish();
    while (!def.finished()) {
      int len=def.deflate(buf,0,buf.length);
      if (len <= 0)       break;
      out.write(buf,0,len);
    }
    if (!def.finished())     throw new Error("Can't deflate all input?");
    out.flush();
  }

   public void close() throws IOException {
    finish();
    out.close();
  }

   public void write(  int bval) throws IOException {
    byte[] b=new byte[1];
    b[0]=(byte)bval;
    write(b,0,1);
  }

   public void write(  byte[] buf) throws IOException {
    write(buf,0,buf.length);
  }

   public void write(  byte[] buf,  int off,  int len) throws IOException {
    def.setInput(buf,off,len);
    deflate();
  }
}
